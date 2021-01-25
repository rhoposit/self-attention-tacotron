import tensorflow as tf
import sys
from tacotron2.tacotron.modules import Embedding
from tacotron2.tacotron.tacotron_v2 import PostNetV2, EncoderV2
from tacotron2.tacotron.hooks import MetricsSaver
from tacotron2.tacotron.losses import codes_loss, classification_loss, binary_loss
from modules.module import ZoneoutEncoderV1, ExtendedDecoder, EncoderV1WithAccentType, \
    SelfAttentionCBHGEncoder, DualSourceDecoder, TransformerDecoder, \
    DualSourceTransformerDecoder, SelfAttentionCBHGEncoderWithAccentType, \
    MgcLf0Decoder, MgcLf0DualSourceDecoder, DualSourceMgcLf0TransformerDecoder
from modules.metrics import MgcLf0MetricsSaver
from modules.regularizers import l2_regularization_loss
from models.attention_factories import attention_factory, dual_source_attention_factory, \
    force_alignment_attention_factory, \
    force_alignment_dual_source_attention_factory
from multi_speaker_tacotron.modules.external_embedding import ExternalEmbedding
from multi_speaker_tacotron.modules.multi_speaker_postnet import MultiSpeakerPostNet


class DualSourceSelfAttentionTacotronModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim)

            if params.use_accent_type:
                accent_embedding = Embedding(params.num_accent_type,
                                             embedding_dim=params.accent_type_embedding_dim,
                                             index_offset=params.accent_type_offset)

            encoder = encoder_factory(params, is_training)

            assert params.decoder in ["DualSourceDecoder", "DualSourceTransformerDecoder"]
            decoder = decoder_factory(params)

            ## make sure that only one of (external_speaker_embedding, speaker_embedding) has been chosen
            assert not (params.use_speaker_embedding and params.use_external_speaker_embedding)
            
            if params.use_speaker_embedding:
                speaker_embedding = Embedding(params.num_speakers,
                                              embedding_dim=params.speaker_embedding_dim,
                                              index_offset=params.speaker_embedding_offset)
            elif params.use_external_speaker_embedding:
                speaker_embedding = ExternalEmbedding(params.embedding_file, params.num_speakers,
                                                      embedding_dim=params.speaker_embedding_dim,
                                                      index_offset=params.speaker_embedding_offset)

            target = labels.codes if (is_training or is_validation) else features.codes

            embedding_output = embedding(features.source)
            encoder_lstm_output, encoder_self_attention_output, self_attention_alignment = encoder(
                (embedding_output, accent_embedding(features.accent_type)),
                input_lengths=features.source_length) if params.use_accent_type else encoder(
                embedding_output, input_lengths=features.source_length)

            ## choose a speaker ID to synthesize as
            x = params.speaker_for_synthesis
            if x > -1:
                speaker_embedding_output = speaker_embedding(x)
            else:  ## default is to just use the speaker ID associated with the test utterance
                speaker_embedding_output = speaker_embedding(
                    features.speaker_id) if params.use_speaker_embedding or params.use_external_speaker_embedding else None

            ## resize speaker embedding with a projection layer
            if params.speaker_embedding_projection_out_dim > -1:
                resize = tf.layers.Dense(params.speaker_embedding_projection_out_dim, activation=tf.nn.relu)
                speaker_embedding_output = resize(speaker_embedding_output)

            ## concatenate encoder outputs with speaker embedding along the time axis
            if params.speaker_embedd_to_decoder:
                expand_speaker_embedding_output = tf.tile(tf.expand_dims(speaker_embedding_output, axis=1),
                                                          [1, tf.shape(encoder_lstm_output)[1], 1])
                encoder_lstm_output = tf.concat((encoder_lstm_output, expand_speaker_embedding_output), axis=-1)
                encoder_self_attention_output = tf.concat(
                    (encoder_self_attention_output, expand_speaker_embedding_output), axis=-1)
                
            attention1_fn, attention2_fn = dual_source_attention_factory(params)
            code_output_raw, stop_token, decoder_state = decoder((encoder_lstm_output, encoder_self_attention_output),
                                                            attention1_fn=attention1_fn,
                                                            attention2_fn=attention2_fn,
                                                            speaker_embed=speaker_embedding_output if params.speaker_embedd_to_prenet else None,
                                                            is_training=is_training,
                                                            is_validation=is_validation or params.use_forced_alignment_mode,
                                                            teacher_forcing=params.use_forced_alignment_mode,
                                                            memory_sequence_length=features.source_length,
                                                            memory2_sequence_length=features.source_length,
                                                            target_sequence_length=labels.target_length if is_training else None,
                                                            target=target,
                                                            apply_dropout_on_inference=params.apply_dropout_on_inference)
#            code_output_raw = tf.Print(code_output_raw, [code_output_raw[3][0]], "\n* code output raw", summarize=-1)
            code_output_softmax = tf.nn.softmax(code_output_raw, axis=None, name=None, dim=None)
#            code_output_softmax = tf.Print(code_output_softmax, [code_output_softmax[3][0]], "\n* code output softmax", summarize=-1)
            code_output = tf.one_hot(tf.argmax(code_output_softmax, axis=2), depth = 512)
#            code_output = tf.Print(code_output, [code_output[3][0]], "\n* code output onehot", summarize=-1)

            # arrange to (B, T_memory, T_query)
            self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in self_attention_alignment]
            if params.decoder == "DualSourceTransformerDecoder" and not is_training:
                decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
                alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                    decoder_state.alignments]
            else:
                decoder_rnn_state = decoder_state[0]
                alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_forced_alignment_mode:
                attention1_fn, attention2_fn = force_alignment_dual_source_attention_factory(params)
                code_output_raw, stop_token, decoder_state = decoder((encoder_lstm_output, encoder_self_attention_output),
                                                                attention1_fn=attention1_fn,
                                                                attention2_fn=attention2_fn,
                                                                speaker_embed=speaker_embedding_output if params.speaker_embedd_to_prenet else None,
                                                                is_training=is_training,
                                                                is_validation=True,
                                                                teacher_forcing=False,
                                                                memory_sequence_length=features.source_length,
                                                                memory2_sequence_length=features.source_length,
                                                                target_sequence_length=labels.target_length if is_training else None,
                                                                target=target,
                                                                teacher_alignments=(
                                                                    tf.transpose(alignment1, perm=[0, 2, 1]),
                                                                    tf.transpose(alignment2, perm=[0, 2, 1])),
                                                                apply_dropout_on_inference=params.apply_dropout_on_inference)
                code_output_softmax = tf.nn.softmax(code_output_raw, axis=None, name=None, dim=None)
                code_output = tf.one_hot(tf.argmax(code_output_softmax, axis=2), depth = 512)

                if params.decoder == "DualSourceTransformerDecoder" and not is_training:
                    alignment1 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[0].stack(),
                                              [1, 2, 0])
                    alignment2 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[1].stack(),
                                              [1, 2, 0])
                    decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                        decoder_state.alignments]
                else:
                    alignment1 = tf.transpose(decoder_state[0].alignment_history[0].stack(), [1, 2, 0])
                    alignment2 = tf.transpose(decoder_state[0].alignment_history[1].stack(), [1, 2, 0])
                    decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time


            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
#                code_output = tf.Print(code_output, [code_output[3][0]], "\ncode output", summarize=-1)
#                out2 = tf.Print(labels.codes, [tf.shape(labels.codes)], "\nlabels.codes")
#                out3 = tf.Print(labels.code_loss_mask, [tf.shape(labels.code_loss_mask)], "\nlabels.code_mask")
#
#                code_loss = codes_loss(code_output, out2, out3,params.code_loss_type)
                code_loss = 0.1*codes_loss(code_output_raw, labels.codes, labels.code_loss_mask, params.code_loss_type)
                
                # fixing labels.done, labels.binary_loss_mask
#                stop_token = tf.Print(stop_token, [tf.shape(stop_token)], "stop_token")
                done_loss = binary_loss(stop_token, labels.done, labels.binary_loss_mask)

                blacklist = ["embedding", "bias", "batch_normalization", "output_projection_wrapper/kernel",
                             "lstm_cell",
                             "output_and_stop_token_wrapper/dense/", "output_and_stop_token_wrapper/dense_1/",
                             "stop_token_projection/kernel"]
                regularization_loss = l2_regularization_loss(
                    tf.trainable_variables(), params.l2_regularization_weight,
                    blacklist) if params.use_l2_regularization else 0

                loss = code_loss + done_loss + regularization_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step,
                    params.learning_rate_step_factor) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(loss, code_loss, done_loss, lr, regularization_loss)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = MetricsSaver([alignment1, alignment2] + self_attention_alignment, global_step,
                                                   code_output, labels.codes,
                                                   labels.target_length,
                                                   features.id,
                                                   features.text,
                                                   params.alignment_save_steps,
                                                   mode, summary_writer,
                                                   save_training_time_metrics=params.save_training_time_metrics,
                                                   keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                    hooks = [alignment_saver]
                    if params.record_profile:
                        profileHook = tf.train.ProfilerHook(save_steps=params.profile_steps, output_dir=model_dir,
                                                            show_dataflow=True, show_memory=True)
                        hooks.append(profileHook)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=hooks)

            if is_validation:
                # validation with teacher forcing
                attention1_fn, attention2_fn = dual_source_attention_factory(params)
                code_output_raw_with_teacher, stop_token_with_teacher, decoder_state_with_teacher = decoder(
                    (encoder_lstm_output, encoder_self_attention_output),
                    attention1_fn=attention1_fn,
                    attention2_fn=attention2_fn,
                    speaker_embed=speaker_embedding_output if params.speaker_embedd_to_prenet else None,
                    is_training=is_training,
                    is_validation=is_validation,
                    memory_sequence_length=features.source_length,
                    memory2_sequence_length=features.source_length,
                    target_sequence_length=labels.target_length,
                    target=target,
                    teacher_forcing=True,
                    apply_dropout_on_inference=params.apply_dropout_on_inference)
                
                code_output_softmax_with_teacher = tf.nn.softmax(code_output_raw_with_teacher, axis=None, name=None, dim=None)
                code_output_with_teacher = tf.one_hot(tf.argmax(code_output_softmax_with_teacher, axis=2), depth = 512)
                code_loss_with_teacher = 0.1*codes_loss(code_output_raw_with_teacher, labels.codes,
                                                  labels.code_loss_mask, params.code_loss_type)
                done_loss_with_teacher = binary_loss(stop_token_with_teacher, labels.done, labels.binary_loss_mask)
                loss_with_teacher = code_loss_with_teacher + done_loss_with_teacher + regularization_loss

                eval_metric_ops = self.get_validation_metrics(code_loss, done_loss,
                                                              loss_with_teacher,
                                                              code_loss_with_teacher, done_loss_with_teacher,
                                                              regularization_loss)

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = MetricsSaver(
                    [alignment1, alignment2] + self_attention_alignment + decoder_self_attention_alignment, global_step,
                    code_output, labels.codes,
                    labels.target_length,
                    features.id,
                    features.text,
                    1,
                    mode, summary_writer,
                    save_training_time_metrics=params.save_training_time_metrics,
                    keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                num_self_alignments = len(self_attention_alignment)
                num_decoder_self_alignments = len(decoder_self_attention_alignment)
                # transform the codes softmax back into onehot
                predictions = {
                    "id": features.id,
                    "key": features.key,
                    "codes": code_output,
                    "ground_truth_codes": features.codes,
                    "alignment": alignment1,
                    "alignment2": alignment2,
                    "alignment3": decoder_self_attention_alignment[0] if num_decoder_self_alignments >= 1 else None,
                    "alignment4": decoder_self_attention_alignment[1] if num_decoder_self_alignments >= 2 else None,
                    "alignment5": self_attention_alignment[0] if num_self_alignments >= 1 else None,
                    "alignment6": self_attention_alignment[1] if num_self_alignments >= 2 else None,
                    "alignment7": self_attention_alignment[2] if num_self_alignments >= 3 else None,
                    "alignment8": self_attention_alignment[3] if num_self_alignments >= 4 else None,
                    "source": features.source,
                    "text": features.text,
                    "accent_type": features.accent_type if params.use_accent_type else None,
                }
                predictions = dict(filter(lambda xy: xy[1] is not None, predictions.items()))
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        super(DualSourceSelfAttentionTacotronModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def learning_rate_decay(init_rate, global_step, step_factor):
        warmup_steps = 4000.0
        step = tf.to_float(global_step * step_factor + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(loss, code_loss, done_loss, learning_rate, l2_regularization_loss):
        if loss is not None:
            tf.summary.scalar("loss_with_teacher", loss)
        if code_loss is not None:
            tf.summary.scalar("code_loss", code_loss)
            tf.summary.scalar("code_loss_with_teacher", code_loss)
        if done_loss is not None:
            tf.summary.scalar("done_loss", done_loss)
            tf.summary.scalar("done_loss_with_teacher", done_loss)
        if l2_regularization_loss is not None:
            tf.summary.scalar("l2_regularization_loss", l2_regularization_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(code_loss, done_loss, loss_with_teacher, code_loss_with_teacher,
                               done_loss_with_teacher, l2_regularization_loss):
        metrics = {}
        if code_loss is not None:
            metrics["code_loss"] = tf.metrics.mean(code_loss)
        if done_loss is not None:
            metrics["done_loss"] = tf.metrics.mean(done_loss)
        if loss_with_teacher is not None:
            metrics["loss_with_teacher"] = tf.metrics.mean(loss_with_teacher)
        if code_loss_with_teacher is not None:
            metrics["code_loss_with_teacher"] = tf.metrics.mean(code_loss_with_teacher)
        if done_loss_with_teacher is not None:
            metrics["done_loss_with_teacher"] = tf.metrics.mean(done_loss_with_teacher)
        if l2_regularization_loss is not None:
            metrics["l2_regularization_loss"] = tf.metrics.mean(l2_regularization_loss)
        return metrics




def encoder_factory(params, is_training):
    if params.encoder == "SelfAttentionCBHGEncoder":
        encoder = SelfAttentionCBHGEncoder(is_training,
                                           cbhg_out_units=params.cbhg_out_units,
                                           conv_channels=params.conv_channels,
                                           max_filter_width=params.max_filter_width,
                                           projection1_out_channels=params.projection1_out_channels,
                                           projection2_out_channels=params.projection2_out_channels,
                                           num_highway=params.num_highway,
                                           self_attention_out_units=params.self_attention_out_units,
                                           self_attention_num_heads=params.self_attention_num_heads,
                                           self_attention_num_hop=params.self_attention_num_hop,
                                           self_attention_transformer_num_conv_layers=params.self_attention_transformer_num_conv_layers,
                                           self_attention_transformer_kernel_size=params.self_attention_transformer_kernel_size,
                                           prenet_out_units=params.encoder_prenet_out_units,
                                           drop_rate=params.encoder_prenet_drop_rate,
                                           zoneout_factor_cell=params.zoneout_factor_cell,
                                           zoneout_factor_output=params.zoneout_factor_output,
                                           self_attention_drop_rate=params.self_attention_drop_rate)
    else:
        raise ValueError(f"Unknown encoder: {params.encoder}")
    return encoder


def decoder_factory(params):
    if params.decoder == "DualSourceTransformerDecoder":
        decoder = DualSourceTransformerDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                               drop_rate=params.decoder_prenet_drop_rate,
                                               attention_rnn_out_units=params.attention_out_units,
                                               decoder_version=params.decoder_version,
                                               decoder_out_units=params.decoder_out_units,
                                               num_mels=512,
                                               outputs_per_step=params.outputs_per_step,
                                               max_iters=params.max_iters,
                                               n_feed_frame=params.n_feed_frame,
                                               zoneout_factor_cell=params.zoneout_factor_cell,
                                               zoneout_factor_output=params.zoneout_factor_output,
                                               self_attention_out_units=params.decoder_self_attention_out_units,
                                               self_attention_num_heads=params.decoder_self_attention_num_heads,
                                               self_attention_num_hop=params.decoder_self_attention_num_hop,
                                               self_attention_drop_rate=params.decoder_self_attention_drop_rate)
    else:
        raise ValueError(f"Unknown decoder: {params.decoder}")
    return decoder


def tacotron_model_factory(hparams, model_dir, run_config, warm_start_from=None):
    if hparams.tacotron_model == "DualSourceSelfAttentionTacotronModel":
        model = DualSourceSelfAttentionTacotronModel(hparams, model_dir,
                                                     config=run_config,
                                                     warm_start_from=warm_start_from)
    else:
        raise ValueError(f"Unknown Tacotron model: {hparams.tacotron_model}")
    return model
