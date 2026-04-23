# -*- coding: utf-8 -*-
#/usr/bin/python3
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processing"))

import tensorflow as tf
# from new_data_loader import *
from model_invoked import Transformer
from  extract_data_subword import *
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
from hparams import Hparams
import math
import logging
import time
from datetime import timedelta

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)
os.environ['CUDA_VISIBLE_DEVICES'] = hp.gpu
outfile = hp.res_log

def run_epoch(session, model, state, summary_writer, epoch=None):
    total_loss = 0.0
    epoch_start_time = time.time()

    data_loader = model.data.batch_iter(hp.batch_size, state, epoch=epoch)
    step = 0
    while True:
        feed_dict = {}

        body_batch, pro_batch, doc_batch, dec_inp_batch, dec_tgt_batch, invoked_batch, batch_len = next(data_loader)
        feed_dict[model.body_batch] = body_batch
        feed_dict[model.pro_batch] = pro_batch
        feed_dict[model.doc_batch] = doc_batch
        feed_dict[model.invoked_batch] = invoked_batch
        feed_dict[model.dec_inp_batch] = dec_inp_batch
        feed_dict[model.dec_tgt_batch] = dec_tgt_batch

        _, _gs, _summary, _loss, _preds = session.run([model.train_op, model.global_step, model.train_summaries, model.loss, model.preds], feed_dict)
        summary_writer.add_summary(_summary, _gs)

        if step % (batch_len // 10) == 10:
            print("%.2f perplexity : %.3f " %
                  (step * 1.0 / batch_len, _loss))

        total_loss += _loss
        step += 1
        if step >= batch_len:
            break

    epoch_elapsed = time.time() - epoch_start_time
    return total_loss / batch_len, _gs, _preds, dec_tgt_batch, epoch_elapsed


def train():
    resout = open(outfile, 'a')

    logging.info("# Load model")
    m = Transformer(hp)
    m.loss, m.train_op, m.global_step, m.train_summaries, m.preds = m.train()
    y_hat = m.eval()

    logging.info("# Session")
    saver = tf.train.Saver(max_to_keep=hp.save_epochs)

    total_start_time = time.time()
    epoch_times = []

    print("=" * 70)
    print("  GTNM Training Started")
    print("  Total Epochs: {}".format(hp.num_epochs))
    print("  Batch Size: {}".format(hp.batch_size))
    print("  Using Project Context (--pro): {}".format(hp.pro))
    print("=" * 70)
    print()

    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(hp.logdir)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.global_variables_initializer())
            save_variable_specs(os.path.join(hp.logdir, "specs"))
        else:
            saver.restore(sess, ckpt)
            print("[RESUME] Restored from checkpoint: {}".format(ckpt))

        summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
        best_eval_val_f1 = 0
        for i in range(hp.num_epochs):
            epoch_start = time.time()

            print("-" * 70)
            print("[Epoch {}/{}] Training...".format(i + 1, hp.num_epochs))
            print("-" * 70)

            train_loss, _global_step, _preds, _tgt, epoch_elapsed = run_epoch(sess, m, 'train_subword', summary_writer, epoch=i)

            _, train_precision, train_recall, train_f1, train_acc = get_hypotheses('train_subword', hp, sess, m, y_hat, m.data.w2id, m.data.id2w, epoch=i)
            print('[Epoch {}] Train - Loss: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} | Acc: {:.4f}'.format(
                i+1, train_loss, train_precision, train_recall, train_f1, train_acc))
            print('[Epoch {}] Train Time: {}'.format(i+1, format_duration(epoch_elapsed)))

            logging.info("# validation")

            logging.info("# get hypotheses")

            eval_start = time.time()
            hypotheses, val_precision, val_recall, val_f1, val_acc = get_hypotheses('eval_subword', hp, sess, m, y_hat, m.data.w2id, m.data.id2w)
            eval_elapsed = time.time() - eval_start

            print('[Epoch {}] Eval  - Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} | Acc: {:.4f}'.format(
                i+1, val_precision, val_recall, val_f1, val_acc))
            print('[Epoch {}] Eval Time:  {}'.format(i+1, format_duration(eval_elapsed)))

            print('[Epoch {}] Total Epoch Time: {}'.format(i+1, format_duration(epoch_elapsed + eval_elapsed)))

            print('epoch {}: eval precision {}, eval recall {}, eval f1 {}, eval acc {}'.format(i+1, val_precision, val_recall, val_f1, val_acc), file=resout)
            resout.flush()

            if val_f1 > best_eval_val_f1:
                best_eval_val_f1 = val_f1
                logging.info("# write eval results")
                model_output = "java_E%02dL%.2f" % (i, train_loss)

                logging.info("# save models")
                ckpt_name = os.path.join(hp.logdir, model_output)
                saver.save(sess, ckpt_name, global_step=_global_step)
                print('[SAVE] New best model! F1 improved to {:.4f}'.format(val_f1))
                print('[SAVE] Model saved to: {}'.format(ckpt_name))

            epoch_times.append(epoch_elapsed + eval_elapsed)

            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = hp.num_epochs - (i + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = format_duration(eta_seconds)

            total_elapsed = time.time() - total_start_time

            print('-' * 70)
            print('[PROGRESS] Epoch {}/{} completed'.format(i + 1, hp.num_epochs))
            print('[PROGRESS] Avg Epoch Time: {} | Total Elapsed: {}'.format(format_duration(avg_epoch_time), format_duration(total_elapsed)))
            print('[PROGRESS] Estimated Time Remaining: ~{} ({} epochs left)'.format(eta_str, remaining_epochs))
            print('[PROGRESS] Best Eval F1 so far: {:.4f}'.format(best_eval_val_f1))
            print('-' * 70)
            print()

            logging.info("# fall back to train mode")

    total_elapsed = time.time() - total_start_time
    print()
    print("=" * 70)
    print("  TRAINING COMPLETED!")
    print("=" * 70)
    print("  Total Training Time: {}".format(format_duration(total_elapsed)))
    print("  Total Epochs: {}".format(hp.num_epochs))
    print("  Average Time per Epoch: {}".format(format_duration(sum(epoch_times) / len(epoch_times))))
    print("  Best Eval F1: {:.4f}".format(best_eval_val_f1))
    print("  Models saved to: {}".format(hp.logdir))
    print("=" * 70)

    summary_writer.close()
    resout.close()


def format_duration(seconds):
    """Format seconds into human-readable duration string"""
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return "{}m {}s".format(minutes, secs)
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return "{}h {}m {}s".format(hours, minutes, secs)

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()


