# Main method for running all tests.
#
# https://github.com/Johan-Verschoor/CL-XD-ABSA/
#
# Adapted from Knoester, Frasincar, and Trușcă (2022)
# https://doi.org/10.1007/978-3-031-20891-1_3
#
# Knoester, J., Frasincar, F., and Trușcă, M. M. (2022). Domain adversarial training for aspect-
# based sentiment analysis. In 22nd International Conference on Web Information Systems
# Engineering (WISE 2022), volume 13724 of LNCS, pages 21–37. Springer.
#
# Originally from:
# Trușcă, M. M., Wassenberg, D., Frasincar, F., and Dekker, R. (2020). A Hybrid Approach for
# Aspect-Based Sentiment Analysis Using Deep Contextual Word Embeddings and Hierarchical Attention.
# In: Bielikova, M., Mikkonen, T., Pautasso, C. (eds) Web Engineering. ICWE 2020.
# Lecture Notes in Computer Science, vol 12128. Springer, Cham.
# https://doi.org/10.1007/978-3-030-50578-3_25


import time
import datetime

import nltk

from config import *
from load_data import *
import HCL_DAT_LCR_Rot_hop

nltk.download('punkt')


def main(_):
    """
    Runs all specified tests.

    :param _:
    :return:
    """
    # After running: back-up results file and model in case of running the model to be saved.
    # It is recommended to turn on logging of the output and to back that up as well for debugging purposes.

    #For CL-XD-ABSA extensions: Fill in the contrastive loss parameters in the DAT_LCR_Rot_hop_plus_plus main function

    rest_lapt = False  # Run restaurant-laptop model.   
    rest_book = False  # Run restaurant-book model.
    lapt_rest = False  # Run laptop-restaurant model.
    lapt_book = False  # Run laptop-book model.
    book_rest = True  # Run book-restaurant model.
    book_lapt = False  # Run book-laptop model.
    write_result = True  # Write results to text file.

    n_iter = 25  # Number of iterations.
    FLAGS.n_iter = n_iter

    # For each model specify the desired batch sizes in order to have equal number of batches for both source and
    # target domain. These numbers can be adjusted to your own will, but take into account the ratio. Afterwards run the
    # neural network with the optimal hyperparameter values.
    if rest_lapt:
        FLAGS.batch_size_src = 24   
        FLAGS.batch_size_tar = 15
        FLAGS.batch_size_te = 701
        # m = 1
        # run_CLRH(source_domain="restaurant", target_domain="laptop", year_source=2014, year_target=2014,
        #     learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        #     l2_dis=0.001, l2_f=0.0001, balance_lambda=0.6, tau_d=0.07, tau_c=0.05, lambda_dcl=0.05, lambda_ccl=0.7, write_result=write_result)
        # m = 2 (CHANGE IN HCL CLASS) with new hp in arch
        # run_CLRH(source_domain="restaurant", target_domain="laptop", year_source=2014, year_target=2014,
        #     learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        #     l2_dis=0.001, l2_f=0.0001, balance_lambda=0.6,tau_d=0.07, tau_c=0.05, lambda_dcl=0.01, lambda_ccl=0.5, write_result=write_result)
        # m = 2 domain ablation
        # run_CLRH(source_domain="restaurant", target_domain="laptop", year_source=2014, year_target=2014,
        #     learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        #     l2_dis=0.001, l2_f=0.0001, balance_lambda=0.6,tau_d=0.07, tau_c=0.05, lambda_dcl=0.0, lambda_ccl=0.5, write_result=write_result)
        # # m = 3
        # m=1 hypers below, now test
        # run_CLRH(source_domain="restaurant", target_domain="laptop", year_source=2014, year_target=2014,
        #     learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        #     l2_dis=0.001, l2_f=0.0001, balance_lambda=0.6, tau_d=0.07, tau_c=0.05, lambda_dcl=0.05, lambda_ccl=0.7, write_result=write_result)
        # Using optimal hypers for m = 2
        run_CLRH(source_domain="restaurant", target_domain="laptop", year_source=2014, year_target=2014,
            learning_rate_dis=0.03, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        # class ablation
            l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6, tau_d=0.05, tau_c=0.1, lambda_dcl=0.2, lambda_ccl=0.0, write_result=write_result)
        # optimal hypers, have low acc,
        # run_CLRH(source_domain="restaurant", target_domain="laptop", year_source=2014, year_target=2014,
        #     learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.85, momentum_f=0.8,
        #     l2_dis=0.01, l2_f=0.001, balance_lambda=0.6, tau_d=0.07, tau_c=0.05, lambda_dcl=0.1, lambda_ccl=0.5, write_result=write_result)
        # domain ablation
        # run_CLRH(source_domain="restaurant", target_domain="laptop", year_source=2014, year_target=2014,
        #         learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.85, momentum_f=0.8,
        #         l2_dis=0.01, l2_f=0.001, balance_lambda=0.6, tau_d=0.07, tau_c=0.05, lambda_dcl=0.1, lambda_ccl=0.0, write_result=write_result)
    if rest_book:
        FLAGS.batch_size_src = 24
        FLAGS.batch_size_tar = 18
        FLAGS.batch_size_te = 804
        # m = 1
        # run_CLRH(source_domain="restaurant", target_domain="book", year_source=2014, year_target=2019,
        #          learning_rate_dis=0.03, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.90, momentum_f=0.80,
        #          l2_dis=0.0001, l2_f=0.001, balance_lambda=1.1,tau_d=0.2, tau_c=0.2, lambda_dcl=0.05, lambda_ccl=0.2, write_result=write_result)
        # # m = 2
        run_CLRH(source_domain="restaurant", target_domain="book", year_source=2014, year_target=2019,
                 learning_rate_dis=0.03, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.90, momentum_f=0.80,
                 l2_dis=0.0001, l2_f=0.001, balance_lambda=1.1,tau_d=0.2, tau_c=0.2, lambda_dcl=0.05, lambda_ccl=0.0, write_result=write_result)
        # m = 3
        #run_CLRH(source_domain="restaurant", target_domain="book", year_source=2014, year_target=2019,
                #  learning_rate_dis=0.03, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.90, momentum_f=0.80,
                #  l2_dis=0.0001, l2_f=0.001, balance_lambda=1.1,tau_d=0.2, tau_c=0.2, lambda_dcl=0.05, lambda_ccl=0.2, write_result=write_result)
    if lapt_rest:
        FLAGS.batch_size_src = 15
        FLAGS.batch_size_tar = 24
        FLAGS.batch_size_te = 1122
        # m = 1
        # run_CLRH(source_domain="laptop", target_domain="restaurant", year_source=2014, year_target=2014,
        #          learning_rate_dis=0.03, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.90, momentum_f=0.80,
        #          l2_dis=0.001, l2_f=0.01, balance_lambda=1.1,
        #          tau_d=0.07, tau_c=0.05, lambda_dcl=0.5, lambda_ccl=0.7, write_result=write_result)
        # m = 2
        # run_CLRH(source_domain="laptop", target_domain="restaurant", year_source=2014, year_target=2014,
        #          learning_rate_dis=0.03, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.80, momentum_f=0.90,
        #          l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,
        #          tau_d=0.1, tau_c=0.1, lambda_dcl=0.2, lambda_ccl=0.2, write_result=write_result)
        # domain ablation
        # run_CLRH(source_domain="laptop", target_domain="restaurant", year_source=2014, year_target=2014,
        # learning_rate_dis=0.03, learning_rate_f=0.05, keep_prob=0.3, momentum_dis=0.80, momentum_f=0.90,
        # l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,
        # tau_d=0.1, tau_c=0.1, lambda_dcl=0.02, lambda_ccl=0.0, write_result=write_result)
        # # m = 3
        # new hyper
        # run_CLRH(source_domain="laptop", target_domain="restaurant", year_source=2014, year_target=2014,
        #          learning_rate_dis=0.03, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.80, momentum_f=0.90,
        #          l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,
        #          tau_d=0.1, tau_c=0.1, lambda_dcl=0.2, lambda_ccl=0.2, write_result=write_result)
        run_CLRH(source_domain="laptop", target_domain="restaurant", year_source=2014, year_target=2014,
                 learning_rate_dis=0.03, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.80, momentum_f=0.90,
                 l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,
                 tau_d=0.1, tau_c=0.1, lambda_dcl=0.05, lambda_ccl=0.0, write_result=write_result)
    if lapt_book:
        FLAGS.batch_size_src = 20
        FLAGS.batch_size_tar = 24
        FLAGS.batch_size_te = 804
        # m = 1 (domain ablation for 2)
        # run_CLRH(source_domain="laptop", target_domain="book", year_source=2014, year_target=2019,
        #          learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.85, momentum_f=0.80,
        #          l2_dis=0.01, l2_f=0.001, balance_lambda=0.6, tau_d=0.2, tau_c=0.1, lambda_dcl=0.5, lambda_ccl=0.0,write_result=write_result)
        # m = 2
        # run_CLRH(source_domain="laptop", target_domain="book", year_source=2014, year_target=2019,
        #          learning_rate_dis=0.005, learning_rate_f=0.03, keep_prob=0.25, momentum_dis=0.8, momentum_f=0.80,
        #          l2_dis=0.0001, l2_f=0.01, balance_lambda=0.8, tau_d=0.07, tau_c=0.05, lambda_dcl=0.1, lambda_ccl=0.5,write_result=write_result)
        # m = 3
        run_CLRH(source_domain="laptop", target_domain="book", year_source=2014, year_target=2019,
                 learning_rate_dis=0.005, learning_rate_f=0.005, keep_prob=0.3, momentum_dis=0.85, momentum_f=0.80,
                 l2_dis=0.01, l2_f=0.001, balance_lambda=0.6, tau_d=0.2, tau_c=0.1, lambda_dcl=0.5, lambda_ccl=0.0,write_result=write_result)
    if book_rest:
        FLAGS.batch_size_src = 18
        FLAGS.batch_size_tar = 24
        FLAGS.batch_size_te = 1122
        # m = 1
        # run_CLRH(source_domain="book", target_domain="restaurant", year_source=2019, year_target=2014,
        #          learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.80, momentum_f=0.85,
        #          l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,tau_d=0.7, tau_c=0.1, lambda_dcl=0.1, lambda_ccl=0.7, write_result=write_result)
        m = 2
        run_CLRH(source_domain="book", target_domain="restaurant", year_source=2019, year_target=2014,
                 learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.80, momentum_f=0.85,
                 l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,tau_d=0.2, tau_c=0.1, lambda_dcl=0.5, lambda_ccl=0.7, write_result=write_result)
        # domain ablation
        # run_CLRH(source_domain="book", target_domain="restaurant", year_source=2019, year_target=2014,
        #          learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.80, momentum_f=0.85,
        #          l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,tau_d=0.2, tau_c=0.1, lambda_dcl=0.5, lambda_ccl=0.0, write_result=write_result)
        # m = 3
        # OG Hyper domain ablation
        # run_CLRH(source_domain="book", target_domain="restaurant", year_source=2019, year_target=2014,
        #          learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.85,
        #          l2_dis=0.0001, l2_f=0.001, balance_lambda=0.6,tau_d=0.07, tau_c=0.1, lambda_dcl=0.1, lambda_ccl=0.0, write_result=write_result)
        # m = 3
        # new Hyper 
        # run_CLRH(source_domain="book", target_domain="restaurant", year_source=2019, year_target=2014,
        #          learning_rate_dis=0.03, learning_rate_f=0.01, keep_prob=0.35, momentum_dis=0.85, momentum_f=0.9,
        #          l2_dis=0.01, l2_f=0.0001, balance_lambda=0.8,tau_d=0.1, tau_c=0.2, lambda_dcl=0.7, lambda_ccl=0.1, write_result=write_result)

    if book_lapt:
        FLAGS.batch_size_src = 24
        FLAGS.batch_size_tar = 20
        FLAGS.batch_size_te = 701
        # m = 1
        # run_CLRH(source_domain="book", target_domain="laptop", year_source=2019, year_target=2014,
        #          learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        #          l2_dis=0.0001, l2_f=0.01, balance_lambda=0.6, tau_d=0.1, tau_c=0.05, lambda_dcl=0.2, lambda_ccl=0.7, write_result=write_result)
        # m = 2
        # run_CLRH(source_domain="book", target_domain="laptop", year_source=2019, year_target=2014,
        #          learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        #          l2_dis=0.0001, l2_f=0.01, balance_lambda=0.6, tau_d=0.1, tau_c=0.05, lambda_dcl=0.2, lambda_ccl=0.7, write_result=write_result)
        # m = 3
        # run_CLRH(source_domain="book", target_domain="laptop", year_source=2019, year_target=2014,
        #          learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
        #          l2_dis=0.0001, l2_f=0.01, balance_lambda=0.6, tau_d=0.1, tau_c=0.05, lambda_dcl=0.2, lambda_ccl=0.7, write_result=write_result)
        run_CLRH(source_domain="book", target_domain="laptop", year_source=2019, year_target=2014,
                 learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3, momentum_dis=0.8, momentum_f=0.9,
                 l2_dis=0.0001, l2_f=0.01, balance_lambda=0.6, tau_d=0.1, tau_c=1, lambda_dcl=0.2, lambda_ccl=0.0, write_result=write_result)
    print('Finished program successfully.')


def run_CLRH(source_domain, target_domain, year_source, year_target, learning_rate_dis, learning_rate_f, keep_prob,
            momentum_dis, momentum_f, l2_dis, l2_f, balance_lambda, tau_d, tau_c, lambda_dcl, lambda_ccl, write_result):
    """
    Run CLRH++ model.

    :param source_domain: source domain name
    :param target_domain: target domain name
    :param year_source: source domain year
    :param year_target: target domain year
    :param learning_rate_dis: learning rate domain discriminator
    :param learning_rate_f: learning rate feature extractor and class discriminator
    :param keep_prob: keep probability
    :param momentum_dis: momentum factor domain discriminator
    :param momentum_f: momemtum factor feature extractor and class discriminator
    :param l2_dis: l2-regularisation term domain discriminator
    :param l2_f: l2-regularisation term feature extractor and class discriminator
    :param write_result: True if the results should be written to the results file
    :return:
    """
    FLAGS.balance_lambda = balance_lambda
    set_hyper_flags(learning_rate_dis, learning_rate_f, keep_prob, momentum_dis, momentum_f, l2_dis, l2_f, tau_d, tau_c, lambda_dcl, lambda_ccl)
    set_other_flags(source_domain=source_domain, source_year=year_source, target_domain=target_domain,
                    target_year=year_target)

    # Create results file
    if write_result:
        with open(FLAGS.results_file, "w") as results:
            results.write(source_domain + " to " + target_domain + "\n---\n")
        FLAGS.writable = 1

    start_time = time.time()

    _, _, _, _, _, _ = load_data_and_embeddings(FLAGS, False)
    
    print('Running HCL')
    _, pred2, fw2, bw2, tl2, tr2 = HCL_DAT_LCR_Rot_hop.main(FLAGS.train_path_source, FLAGS.train_path_target, FLAGS.test_path,
                                            FLAGS.learning_rate_dis, FLAGS.learning_rate_f, FLAGS.keep_prob,
                                             FLAGS.momentum_dis, FLAGS.momentum_f, FLAGS.l2_dis, FLAGS.l2_f, FLAGS.balance_lambda,
                                             FLAGS.tau_d, FLAGS.tau_c, FLAGS.lambda_dcl, FLAGS.lambda_ccl
                                             )
                                    

    end_time = time.time()
    run_time = end_time - start_time
    if write_result:
        with open(FLAGS.results_file, "a") as results:
            results.write("Runtime: " + str(run_time) + " seconds.\n\n")
            results.write("Finish:" + str(datetime.datetime.now()))


def set_hyper_flags(learning_rate_dis, learning_rate_f, keep_prob, momentum_dis, momentum_f, l2_dis, l2_f,tau_d, tau_c, lambda_dcl, lambda_ccl):
    """
    Sets hyperparameter flags.

    :param learning_rate_dis: learning rate domain discriminator
    :param learning_rate_f: learning rate feature extractor and class discriminator
    :param keep_prob: keep probability
    :param momentum_dis: momentum factor domain discriminator
    :param momentum_f: momentum factor feature extractor and class discriminator
    :param l2_dis: l2-regularisation term domain discriminator
    :param l2_f: l2-regularisation term feature extractor and class discriminator
    :return:
    """
    FLAGS.learning_rate_dis = learning_rate_dis
    FLAGS.learning_rate_f = learning_rate_f
    FLAGS.keep_prob = keep_prob
    FLAGS.momentum_dis = momentum_dis
    FLAGS.momentum_f = momentum_f
    FLAGS.l2_dis = l2_dis
    FLAGS.l2_f = l2_f
    FLAGS.tau_d = tau_d
    FLAGS.tau_c = tau_c
    FLAGS.lambda_dcl = lambda_dcl
    FLAGS.lambda_ccl = lambda_ccl


def set_other_flags(source_domain, source_year, target_domain, target_year):
    """
    Set other flags.

    :param source_domain: the source domain
    :param source_year: the year of the source domain dataset
    :param target_domain: the target domain
    :param target_year: the year of the target domain dataset
    :return:
    """
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.source_year = source_year
    FLAGS.target_year = target_year

    FLAGS.train_data_source = "data/externalData/" + FLAGS.source_domain + "_train_" + str(FLAGS.source_year) + ".xml"
    FLAGS.train_data_target = "data/externalData/" + FLAGS.target_domain + "_train_" + str(FLAGS.target_year) + ".xml"
    FLAGS.test_data = "data/externalData/" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + ".xml"

    FLAGS.train_path_source = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(source_year) + "_BERT.txt"
    FLAGS.train_path_target = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_train_" + str(target_year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + "_BERT.txt"

    FLAGS.train_embedding_source = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.train_embedding_target = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    FLAGS.test_embedding = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    FLAGS.prob_file = 'prob_' + str(FLAGS.source_domain) + "_" + str(FLAGS.target_domain)
    FLAGS.results_file = "Result_Files/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.target_year) + "Claas.txt"

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()