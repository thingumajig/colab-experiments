from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from bert import modeling
from bert import tokenization
import tensorflow as tf
import codecs

import bert_bilstm_model as m

def create_estimator(label_list):
  FLAGS = tf.flags.FLAGS

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
      "Cannot use sequence length %d because the BERT model "
      "was only trained up to sequence length %d" %
      (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

  run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    keep_checkpoint_max=FLAGS.keep_checkpoint_max,
    save_summary_steps=FLAGS.save_summary_steps,
    tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.num_tpu_cores,
      per_host_input_for_training=is_per_host))

  model_fn = m.model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list) + 1,  # 1 for '0' padding
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu)

  ws = None
  if os.path.exists(FLAGS.output_dir):
    print(f'================== use WarmStartSettings from {FLAGS.output_dir}')
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=FLAGS.output_dir)
  estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    predict_batch_size=FLAGS.predict_batch_size,
    warm_start_from=ws
  )
  return estimator, tokenizer


def train(processor, estimator, tokenizer, label_list):
  FLAGS = tf.flags.FLAGS

  data_config_path = os.path.join(FLAGS.output_dir, FLAGS.data_config_path)
  if os.path.exists(data_config_path):
    with codecs.open(data_config_path) as fd:
      print(f'=========== load existed config:{FLAGS.data_config_path}')
      data_config = json.load(fd)
  else:
    data_config = {}

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if len(data_config) == 0:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int((len(train_examples) / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    data_config['num_train_steps'] = num_train_steps
    data_config['num_warmup_steps'] = num_warmup_steps
    data_config['num_train_size'] = len(train_examples)
  else:
    num_train_steps = int(data_config['num_train_steps'])
    num_warmup_steps = int(data_config['num_warmup_steps'])

  # prepare train_input_fn
  if data_config.get('train.tf_record_path', '') == '':
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    m.filed_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  else:
    train_file = data_config.get('train.tf_record_path')
  print(f'================== train file: {train_file}')
  num_train_size = num_train_size = int(data_config['num_train_size'])
  print("***** Running training *****")
  print(f"  Num examples = {num_train_size} Batch size = {FLAGS.train_batch_size} Num steps = {num_train_steps}")
  train_input_fn = m.file_based_input_fn_builder(
    input_file=train_file,
    seq_length=FLAGS.max_seq_length,
    is_training=True,
    drop_remainder=True)
  # prepare eval_input_fn
  if data_config.get('eval.tf_record_path', '') == '':
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    m.filed_based_convert_examples_to_features(
      eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    data_config['eval.tf_record_path'] = eval_file
    data_config['num_eval_size'] = len(eval_examples)
  else:
    eval_file = data_config['eval.tf_record_path']
  num_eval_size = data_config.get('num_eval_size', 0)
  print("***** Running evaluation *****")
  print(f"  Num examples = {num_eval_size} Batch size = {FLAGS.eval_batch_size}")
  eval_steps = None
  if FLAGS.use_tpu:
    eval_steps = int(num_eval_size / FLAGS.eval_batch_size)
  eval_drop_remainder = True if FLAGS.use_tpu else False
  eval_input_fn = m.file_based_input_fn_builder(
    input_file=eval_file,
    seq_length=FLAGS.max_seq_length,
    is_training=False,
    drop_remainder=eval_drop_remainder)
  # train and evaluate tf.estimator.experimental.
  hook = tf.estimator.experimental.stop_if_no_decrease_hook(
    estimator, 'eval_f', 3000, min_steps=30000, run_every_secs=120)
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[hook])
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=120)
  tp = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  result = tp[0]

  output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
  with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
      print(f"  {key} = {str(result[key])}")
      writer.write(f"{key} = {str(result[key])}\n")

  if not os.path.exists(data_config_path):
    with codecs.open(data_config_path, 'a', encoding='utf-8') as fd:
      json.dump(data_config, fd)


def predict(processor, estimator, tokenizer, label_list, text):
  FLAGS = tf.flags.FLAGS

  # prepare predict_input_fn
  # predict_examples = processor.get_test_examples(FLAGS.data_dir)
  # predict_examples = processor.get_predict_examples(os.path.join(FLAGS.output_dir, filename))
  predict_examples = processor.get_predict_examples_from_str(text)
  print(f'================= Number of predict examples:{len(predict_examples)}')

  predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
  m.filed_based_convert_examples_to_features(predict_examples, label_list,
                                             FLAGS.max_seq_length, tokenizer,
                                             predict_file, mode="test")

  print("***** Running prediction*****")
  print(f"  Num examples = {len(predict_examples)} Batch size = {FLAGS.predict_batch_size}")
  if FLAGS.use_tpu:
    # Warning: According to tpu_estimator.py Prediction on TPU is an
    # experimental feature and hence not supported here
    raise ValueError("Prediction in TPU not supported")
  predict_drop_remainder = True if FLAGS.use_tpu else False
  predict_input_fn = m.file_based_input_fn_builder(
    input_file=predict_file,
    seq_length=FLAGS.max_seq_length,
    is_training=False,
    drop_remainder=predict_drop_remainder)

  # predict
  result = estimator.predict(input_fn=predict_input_fn)

  return predict_examples, result