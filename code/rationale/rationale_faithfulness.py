
import os, sys, gzip
import time
import math
import json
import pickle

import numpy as np
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn import create_optimization_updates, get_activation_by_name, sigmoid, linear
from nn import EmbeddingLayer, Layer, LSTM, RCNN, apply_dropout, default_rng
from utils import say
import myio
import options
from extended_layers import ExtRCNN, ExtLSTM

class Generator(object):

    def __init__(self, args, embedding_layer, nclasses, encoder):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        self.encoder = encoder

    def ready(self):
        encoder = self.encoder
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = encoder.dropout

        # len*batch
        x = self.x = encoder.x
        z = self.z = encoder.z

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]
        layer_type = args.layer.lower()
        for i in xrange(2):
            if layer_type == "rcnn":
                l = RCNN(
                        n_in = n_e,# if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order
                    )
            elif layer_type == "lstm":
                l = LSTM(
                        n_in = n_e,# if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            layers.append(l)

        # len * batch
        #masks = T.cast(T.neq(x, padding_id), theano.config.floatX)
        masks = T.cast(T.neq(x, padding_id), "int8").dimshuffle((0,1,"x"))

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)

        flipped_embs = embs[::-1]

        # len*bacth*n_d
        h1 = layers[0].forward_all(embs)
        h2 = layers[1].forward_all(flipped_embs)
        h_final = T.concatenate([h1, h2[::-1]], axis=2)
        h_final = apply_dropout(h_final, dropout)
        size = n_d * 2

        output_layer = self.output_layer = Layer(
                n_in = size,
                n_out = 1,
                activation = sigmoid
            )

        # len*batch*1
        probs = output_layer.forward(h_final)

        # len*batch
        probs2 = probs.reshape(x.shape)
        self.MRG_rng = MRG_RandomStreams(seed=123)
        z_pred = self.z_pred = T.cast(self.MRG_rng.binomial(size=probs2.shape, p=probs2), "int8")

        # we are computing approximated gradient by sampling z;
        # so should mark sampled z not part of the gradient propagation path
        #
        self.z_pred = theano.gradient.disconnected_grad(z_pred)

        z2 = z.dimshuffle((0,1,"x"))
        logpz = - T.nnet.binary_crossentropy(probs, z2) * masks
        logpz = self.logpz = logpz.reshape(x.shape)
        probs = self.probs = probs.reshape(x.shape)

        # batch
        zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
        zdiff = T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)

        loss_mat = encoder.loss_mat
        if args.aspect < 0:
            loss_vec = T.mean(loss_mat, axis=1)
        else:
            assert args.aspect < self.nclasses
            loss_vec = loss_mat[:,args.aspect]
        self.loss_vec = loss_vec

        coherent_factor = args.sparsity * args.coherent
        loss = self.loss = T.mean(loss_vec)
        sparsity_cost = self.sparsity_cost = T.mean(zsum) * args.sparsity + \
                                             T.mean(zdiff) * coherent_factor
        cost_vec = loss_vec + zsum * args.sparsity + zdiff * coherent_factor
        cost_logpz = T.mean(cost_vec * T.sum(logpz, axis=0))
        self.obj = T.mean(cost_vec)

        params = self.params = [ ]
        for l in layers + [ output_layer ]:
            for p in l.params:
                params.append(p)
        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                                        for x in params)
        say("total # parameters: {}\n".format(nparams))

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p**2)
            else:
                l2_cost = l2_cost + T.sum(p**2)
        l2_cost = l2_cost * args.l2_reg

        cost = self.cost = cost_logpz * 10 + l2_cost
        # print "cost.dtype", cost.dtype

        self.cost_e = loss * 10 + encoder.l2_cost

class Encoder(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # len*batch
        x = self.x = T.imatrix()

        z = self.z = T.bmatrix()
        z = z.dimshuffle((0,1,"x"))

        # batch*nclasses
        y = self.y = T.fmatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]
        depth = args.depth
        layer_type = args.layer.lower()
        for i in xrange(depth):
            if layer_type == "rcnn":
                l = ExtRCNN(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order
                    )
            elif layer_type == "lstm":
                l = ExtLSTM(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            layers.append(l)

        # len * batch * 1
        masks = T.cast(T.neq(x, padding_id).dimshuffle((0,1,"x")) * z, theano.config.floatX)
        # batch * 1
        cnt_non_padding = T.sum(masks, axis=0) + 1e-8

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)

        pooling = args.pooling
        lst_states = [ ]
        h_prev = embs
        for l in layers:
            # len*batch*n_d
            h_next = l.forward_all(h_prev, z)
            if pooling:
                # batch * n_d
                masked_sum = T.sum(h_next * masks, axis=0)
                lst_states.append(masked_sum/cnt_non_padding) # mean pooling
            else:
                lst_states.append(h_next[-1]) # last state
            h_prev = apply_dropout(h_next, dropout)

        if args.use_all:
            size = depth * n_d
            # batch * size (i.e. n_d*depth)
            h_final = T.concatenate(lst_states, axis=1)
        else:
            size = n_d
            h_final = lst_states[-1]
        h_final = apply_dropout(h_final, dropout)

        output_layer = self.output_layer = Layer(
                n_in = size,
                n_out = self.nclasses,
                activation = sigmoid
            )

        # batch * nclasses
        preds = self.preds = output_layer.forward(h_final)

        preds_mask = preds > 0.5
        
        # batch
        # loss_mat = self.loss_mat = (preds-y)**2
        loss_mat = self.loss_mat = T.nnet.binary_crossentropy(preds, y)
        loss = self.loss = T.mean(loss_mat)

        # pred_diff = self.pred_diff = T.mean(T.max(preds, axis=1) - T.min(preds, axis=1))

        params = self.params = [ ]
        for l in layers + [ output_layer ]:
            for p in l.params:
                params.append(p)
        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                                        for x in params)
        say("total # parameters: {}\n".format(nparams))

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p**2)
            else:
                l2_cost = l2_cost + T.sum(p**2)
        l2_cost = l2_cost * args.l2_reg
        self.l2_cost = l2_cost

        cost = self.cost = loss * 10 + l2_cost
        # print T.sum(T.ones_like(y))
        self.accuracy = T.sum(T.eq(preds_mask, y)) / T.sum(T.ones_like(y))

class Model(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses
        self.encoder = Encoder(args, embedding_layer, nclasses)
        self.generator = Generator(args, embedding_layer, nclasses, self.encoder)
        self.encoder.ready()
        self.generator.ready()
        self.dropout = self.encoder.dropout
        self.x = self.encoder.x
        self.y = self.encoder.y
        self.z = self.encoder.z
        self.z_pred = self.generator.z_pred

    def save_model(self, path, args):
        # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        # output to path
        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.encoder.params ],   # encoder
                 [ x.get_value() for x in self.generator.params ], # generator
                 self.nclasses,
                 args                                 # training configuration
                ),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )


    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            eparams, gparams, nclasses, args  = pickle.load(fin)

        # construct model/network using saved configuration
        self.args = args
        self.nclasses = nclasses
        self.ready()
        for x,v in zip(self.encoder.params, eparams):
            x.set_value(v)
        for x,v in zip(self.generator.params, gparams):
            x.set_value(v)


    def train(self, train, dev, test, test_rat, test_norat, rationale_data):
        args = self.args
        dropout = self.dropout
        padding_id = self.embedding_layer.vocab_map["<padding>"]

        if dev is not None:
            dev_batches_x, dev_batches_y = myio.create_batches(
                            dev[0], dev[1], args.batch, padding_id
                        )
        if test is not None:
            test_batches_x, test_batches_y = myio.create_batches(
                            test[0], test[1], args.batch, padding_id
                        )
            test_rat_batches_x, test_rat_batches_y = myio.create_batches(
                            test_rat[0], test_rat[1], args.batch, padding_id
                        )
            test_norat_batches_x, test_norat_batches_y = myio.create_batches(
                            test_norat[0], test_norat[1], args.batch, padding_id
                        )
        if rationale_data is not None:
            valid_batches_x, valid_batches_y = myio.create_batches(
                    [ u["xids"] for u in rationale_data ],
                    [ u["y"] for u in rationale_data ],
                    args.batch,
                    padding_id,
                    sort = False
                )

        start_time = time.time()
        train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )
        say("{:.2f}s to create training batches\n\n".format(
                time.time()-start_time
            ))

        updates_e, lr_e, gnorm_e = create_optimization_updates(
                               cost = self.generator.cost_e,
                               params = self.encoder.params,
                               method = args.learning,
                               lr = args.learning_rate
                        )[:3]


        updates_g, lr_g, gnorm_g = create_optimization_updates(
                               cost = self.generator.cost,
                               params = self.generator.params,
                               method = args.learning,
                               lr = args.learning_rate
                        )[:3]

        sample_generator = theano.function(
                inputs = [ self.x ],
                outputs = self.z_pred,
                #updates = self.generator.sample_updates
                # allow_input_downcast = True
            )

        get_loss_and_pred = theano.function(
                inputs = [ self.x, self.z, self.y ],
                outputs = [ self.generator.loss_vec, self.encoder.preds ]
            )

        eval_generator = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.z, self.generator.obj, self.generator.loss,
                                self.encoder.accuracy, self.encoder.preds ],
                givens = {
                    self.z : self.generator.z_pred
                },
                #updates = self.generator.sample_updates,
                #no_default_updates = True
            )

        train_generator = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.generator.obj, self.generator.loss, \
                                self.generator.sparsity_cost, self.z, gnorm_g, gnorm_e , self.encoder.accuracy],
                givens = {
                    self.z : self.generator.z_pred
                },
                #updates = updates_g,
                updates = updates_g.items() + updates_e.items() #+ self.generator.sample_updates,
                #no_default_updates = True
            )

        eval_period = args.eval_period
        unchanged = 0
        best_dev = 0
        best_dev_e = 1e+2
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        for epoch in xrange(args.max_epochs):
            unchanged += 1
            # if unchanged > 10: break

            train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )

            processed = 0
            train_cost = 0.0
            train_loss = 0.0
            train_sparsity_cost = 0.0
            p1 = 0.0
            train_accuracy = 0.0
            start_time = time.time()

            N = len(train_batches_x)
            for i in xrange(N):
                if (i+1) % 100 == 0:
                    say("\r{}/{}     ".format(i+1,N))

                bx, by = train_batches_x[i], train_batches_y[i]
                mask = bx != padding_id

                cost, loss, sparsity_cost, bz, gl2_g, gl2_e, accuracy = train_generator(bx, by)

                k = len(by)
                processed += k
                train_cost += cost
                train_loss += loss
                train_sparsity_cost += sparsity_cost
                p1 += np.sum(bz*mask) / (np.sum(mask)+1e-8)
                train_accuracy += accuracy

                if (i == N-1) or (eval_period > 0 and processed/eval_period >
                                    (processed-k)/eval_period):
                    say("\n")
                    say(("Generator Epoch {:.2f}  costg={:.4f}  scost={:.4f}  lossg={:.4f}  " +
                        "p[1]={:.2f}  |g|={:.4f} {:.4f}\t[{:.2f}m / {:.2f}m]  accuracy={:.4f}\n ").format(
                            epoch+(i+1.0)/N,
                            train_cost / (i+1),
                            train_sparsity_cost / (i+1),
                            train_loss / (i+1),
                            p1 / (i+1),
                            float(gl2_g),
                            float(gl2_e),
                            (time.time()-start_time)/60.0,
                            (time.time()-start_time)/60.0/(i+1)*N,
                            train_accuracy / (i+1)
                        ))
                    # say("\t"+str([ "{:.1f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                    #                 for x in self.encoder.params ])+"\n")
                    # say("\t"+str([ "{:.1f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                    #                 for x in self.generator.params ])+"\n")

                    if dev:
                        self.dropout.set_value(0.0)
                        dev_obj, dev_loss, dev_acc, dev_p1, preds, lbls = self.evaluate_data(
                                dev_batches_x, dev_batches_y, eval_generator, sampling=True)

                        if dev_acc > best_dev:
                            best_dev = dev_acc
                            unchanged = 0
                            if args.dump and rationale_data:
                                self.dump_rationales(args.dump, valid_batches_x, valid_batches_y,
                                            get_loss_and_pred, sample_generator)

                            if args.save_model:
                                # if not os.path.exists(args.save_model):
                                #     os.makedirs(args.save_model)
                                self.save_model(args.save_model, args)

                        say(("\tsampling devg={:.4f}  mseg={:.4f}" +
                                    "  best_dev={:.4f}  accuracy={:.4f}\n").format(
                            dev_obj,
                            dev_loss,
                            # dev_diff,
                            # dev_p1,
                            best_dev,
                            dev_acc
                        ))

                        if rationale_data is not None:
                            r_mse, r_p1, r_prec1, r_prec2 = self.evaluate_rationale(
                                    rationale_data, valid_batches_x,
                                    valid_batches_y, eval_generator)
                            say(("\trationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
                                        "  prec2={:.4f}\n").format(
                                    r_mse,
                                    r_p1,
                                    r_prec1,
                                    r_prec2
                            ))

                        self.dropout.set_value(dropout_prob)

        if test:
            

            # load best model from training, if stored
            if args.save_model:
                print 'loading best model from training'
                self.load_model(args.save_model)
            
            self.dropout.set_value(0.0)
            test_obj, test_loss, test_acc, test_p1, preds, labels = self.evaluate_data(
                    test_batches_x, test_batches_y, eval_generator, sampling=True)

            test_obj, test_loss, test_acc_rat, test_p1, preds_rat, labels_rat = self.evaluate_data(
                    test_rat_batches_x, test_rat_batches_y, eval_generator, sampling=True)

            test_obj, test_loss, test_acc_norat, test_p1, preds_norat, labels_norat = self.evaluate_data(
                    test_norat_batches_x, test_norat_batches_y, eval_generator, sampling=True)


            # if dev_obj < best_dev:
            #     best_dev = dev_obj
            #     unchanged = 0
            #     if args.dump and rationale_data:
            #         self.dump_rationales(args.dump, valid_batches_x, valid_batches_y,
            #                     get_loss_and_pred, sample_generator)

            #     if args.save_model:
            #         self.save_model(args.save_model, args)

            preds = preds[0].ravel()
            labels = labels[0].ravel()
            preds_rat = preds_rat[0].ravel()
            labels_rat = labels_rat[0].ravel()
            preds_norat = preds_norat[0].ravel()
            labels_norat = labels_norat[0].ravel()

            say(("\ttest accuracy={:.4f}\n").format(
                test_acc
            ))

            say(("\trationale only test accuracy={:.4f}\n").format(
                test_acc_rat
            ))

            say(("\tno rationale test accuracy={:.4f}\n").format(
                test_acc_norat
            ))
            #class_predictions = preds[0] < 0.5
            pred_proba = np.abs(np.ones_like(labels) - labels - preds.ravel()) 
            pred_proba_rat = np.abs(np.ones_like(labels_rat) - labels_rat - preds_rat.ravel())
            pred_proba_norat = np.abs(np.ones_like(labels_norat) - labels_norat - preds_norat.ravel())
            
            say("Sufficiency: {}\n".format((pred_proba - pred_proba_rat).mean()))
            say("Comprehensiveness: {}\n".format((pred_proba - pred_proba_norat).mean()))

            # if rationale_data is not None:
            #     r_mse, r_p1, r_prec1, r_prec2 = self.evaluate_rationale(
            #             rationale_data, valid_batches_x,
            #             valid_batches_y, eval_generator)
            #     say(("\trationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
            #                 "  prec2={:.4f}\n").format(
            #             r_mse,
            #             r_p1,
            #             r_prec1,
            #             r_prec2
            #     ))

            self.dropout.set_value(dropout_prob)



    def evaluate_data(self, batches_x, batches_y, eval_func, sampling=False):
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        tot_obj, tot_mse, tot_diff, p1 = 0.0, 0.0, 0.0, 0.0
        preds = []
        labels = []
        for bx, by in zip(batches_x, batches_y):
            if not sampling:
                e, d = eval_func(bx, by)
            else:
                mask = bx != padding_id
                bz, o, e, d, pred = eval_func(bx, by)
                preds.append(pred)
                labels.append(by)
                p1 += np.sum(bz*mask) / (np.sum(mask) + 1e-8)
                tot_obj += o
            tot_mse += e
            tot_diff += d
        n = len(batches_x)
        if not sampling:
            return tot_mse/n, tot_diff/n
        return tot_obj/n, tot_mse/n, tot_diff/n, p1/n, preds, labels

    def evaluate_rationale(self, reviews, batches_x, batches_y, eval_func):
        args = self.args
        assert args.aspect >= 0
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        aspect = str(args.aspect)
        p1, tot_mse, tot_prec1, tot_prec2 = 0.0, 0.0, 0.0, 0.0
        tot_z, tot_n = 1e-10, 1e-10
        cnt = 0
        for bx, by in zip(batches_x, batches_y):
            mask = bx != padding_id
            bz, o, e, d = eval_func(bx, by)
            tot_mse += e
            p1 += np.sum(bz*mask)/(np.sum(mask) + 1e-8)
            for z,m in zip(bz.T, mask.T):
                z = [ vz for vz,vm in zip(z,m) if vm ]
                assert len(z) == len(reviews[cnt]["xids"])
                truez_intvals = reviews[cnt][aspect]
                prec = sum( 1 for i, zi in enumerate(z) if zi>0 and \
                            any(i>=u[0] and i<u[1] for u in truez_intvals) )
                nz = sum(z)
                if nz > 0:
                    tot_prec1 += prec/(nz+0.0)
                    tot_n += 1
                tot_prec2 += prec
                tot_z += nz
                cnt += 1
        assert cnt == len(reviews)
        n = len(batches_x)
        return tot_mse/n, p1/n, tot_prec1/tot_n, tot_prec2/tot_z

    def dump_rationales(self, path, batches_x, batches_y, eval_func, sample_func):
        embedding_layer = self.embedding_layer
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        lst = [ ]
        for bx, by in zip(batches_x, batches_y):
            bz = np.ones(bx.shape, dtype="int8")
            loss_vec_t, preds_t = eval_func(bx, bz, by)
            bz = sample_func(bx)
            loss_vec_r, preds_r = eval_func(bx, bz, by)
            assert len(loss_vec_r) == bx.shape[1]
            for loss_t, p_t, loss_r, p_r, x,y,z in zip(loss_vec_t, preds_t, \
                                loss_vec_r, preds_r, bx.T, by, bz.T):
                loss_t, loss_r = float(loss_t), float(loss_r)
                p_t, p_r, x, y, z = p_t.tolist(), p_r.tolist(), x.tolist(), y.tolist(), z.tolist()
                w = embedding_layer.map_to_words(x)
                r = [ u if v == 1 else "__" for u,v in zip(w,z) ]
                diff = max(y)-min(y)
                lst.append((diff, loss_t, loss_r, r, w, x, y, z, p_t, p_r))

        #lst = sorted(lst, key=lambda x: (len(x[3]), x[2]))
        with open(path,"w") as fout:
            for diff, loss_t, loss_r, r, w, x, y, z, p_t, p_r in lst:
                fout.write( json.dumps( { "diff": diff,
                                          "loss_t": loss_t,
                                          "loss_r": loss_r,
                                          "rationale": " ".join(r),
                                          "text": " ".join(w),
                                          "x": x,
                                          "z": z,
                                          "y": y,
                                          "p_t": p_t,
                                          "p_r": p_r } ) + "\n" )


def main():
    print args
    assert args.embedding, "Pre-trained word embeddings required."

    embedding_layer = myio.create_embedding_layer(
                        args.embedding
                    )

    max_len = args.max_len

    if args.train:
        train_x, train_y = myio.get_reviews_rational(args.train, data_name='movies', split='train')
        train_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in train_x ]
   

    if args.dev:
        dev_x, dev_y = myio.get_reviews_rational(args.dev, data_name='movies', split='val')
        dev_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in dev_x ]


    if args.test:
        # testx, test_y = myio.read_annotations(args.test)
        test_x, test_y = myio.get_reviews_rational(args.test, data_name='movies', split='test')
        test_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in test_x ]

    if args.test:
        # testx, test_y = myio.read_annotations(args.test)
        test_x_rat, test_y_rat = myio.get_reviews_rational(args.test, data_name='movies', split='test_rat')
        test_x_rat = [ embedding_layer.map_to_ids(x)[:max_len] for x in test_x_rat ]

    if args.test:
        # testx, test_y = myio.read_annotations(args.test)
        test_x_norat, test_y_norat = myio.get_reviews_rational(args.test, data_name='movies', split='test_norat')
        test_x_norat = [ embedding_layer.map_to_ids(x)[:max_len] for x in test_x_norat ]

    if args.load_rationale:
        rationale_data = myio.read_rationales(args.load_rationale)
        for x in rationale_data:
            x["xids"] = embedding_layer.map_to_ids(x["x"])

    if args.train:
        # print len(train_y[0])
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = 1 #len(train_y[0])
                )
        model.ready()

        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                (test_x, test_y),
                (test_x_rat, test_y_rat),
                (test_x_norat, test_y_norat),
                rationale_data if args.load_rationale else None
            )


    if args.load_model and args.dev and not args.train:
        model = Model(
                    args = None,
                    embedding_layer = embedding_layer,
                    nclasses = -1
                )
        model.load_model(args.load_model)
        say("model loaded successfully.\n")

        # compile an evaluation function
        eval_func = theano.function(
                inputs = [ model.x, model.y ],
                outputs = [ model.z, model.generator.obj, model.generator.loss,
                                model.encoder.accuracy, model.encoder.preds ],
                givens = {
                    model.z : model.generator.z_pred
                },
            )

        # compile a predictor function
        pred_func = theano.function(
                inputs = [ model.x ],
                outputs = [ model.z, model.encoder.preds ],
                givens = {
                    model.z : model.generator.z_pred
                },
            )

        # batching data
        padding_id = embedding_layer.vocab_map["<padding>"]
        dev_batches_x, dev_batches_y = myio.create_batches(
                        test_x, test_y, args.batch, padding_id
                    )

        dev_batches_x_rat, dev_batches_y_rat = myio.create_batches(
                        test_x_rat, test_y_rat, args.batch, padding_id
                    )

        dev_batches_x_norat, dev_batches_y_norat = myio.create_batches(
                        test_x_norat, test_y_norat, args.batch, padding_id
                    )

        # disable dropout
        model.dropout.set_value(0.0)
        _, _, acc_full, _, preds, labels = model.evaluate_data(
                dev_batches_x, dev_batches_y, eval_func, sampling=True)
        say("Test Acc: {}\n".format(acc_full))

        _, _, acc_rat, _, preds_rat, labels_rat = model.evaluate_data(
                dev_batches_x_rat, dev_batches_y_rat, eval_func, sampling=True)

        say("Only Rational Test Acc: {}\n".format(acc_rat))

        _, _, acc_norat, _, preds_norat, labels_norat = model.evaluate_data(
                dev_batches_x_norat, dev_batches_y_norat, eval_func, sampling=True)
                
        say("No Rationals Test Acc: {}\n".format(acc_norat))
        preds = preds[0].ravel()
        labels = labels[0].ravel()
        preds_rat = preds_rat[0].ravel()
        labels_rat = labels_rat[0].ravel()
        preds_norat = preds_norat[0].ravel()
        labels_norat = labels_norat[0].ravel()

        pred_proba = np.abs(np.ones_like(labels) - labels - preds) 
        pred_proba_rat = np.abs(np.ones_like(labels_rat) - labels_rat - preds_rat)
        pred_proba_norat = np.abs(np.ones_like(labels_norat) - labels_norat - preds_norat)
            
        say("Sufficiency: {}\n".format((pred_proba - pred_proba_rat).mean()))
        say("Comprehensiveness: {}\n".format((pred_proba - pred_proba_norat).mean()))



if __name__=="__main__":
    args = options.load_arguments()
    main()
