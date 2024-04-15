import logging
from threading import Thread
import os
import psutil
import bz2
import pickle
import numpy
import pickle
import torch
import numpy as np
import torch.multiprocessing as mp
import pandas as pd

import wx

#ML libraries
import gensim
#If not getting updated, use this code https://github.com/maximtrp/bitermplus
import bitermplus as btm
from sklearn.decomposition import NMF as nmf
from sklearn.feature_extraction.text import TfidfVectorizer
from top2vec import Top2Vec
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

import Common.CustomEvents as CustomEvents
import Common.Objects.Utilities.Samples as SamplesUtilities

class CaptureThread(Thread):
    def __init__(self, notify_window, main_frame, model_paramaters, model_type):
        Thread.__init__(self)
        self._notify_window = notify_window
        self.main_frame = main_frame
        self.model_parameters = model_paramaters
        self.model_type = model_type
        self.start()
    
    def run(self):
        dataset_key = self.model_parameters['dataset_key']
        self.model_parameters['tokensets'], field_list = SamplesUtilities.CaptureTokens(dataset_key, self.main_frame)
        wx.PostEvent(self._notify_window, CustomEvents.CaptureResultEvent(self.model_type, self.model_parameters, field_list))


class LDATrainingThread(Thread):
    """LDATrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics, num_passes, alpha, eta):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.num_passes = num_passes
        self.alpha = alpha
        self.eta = eta
        self.start()

    def run(self):
        '''Generates an LDA model'''
        logger = logging.getLogger(__name__+"LDATrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        logger.info("Starting generation of model")
        tokensets_keys = list(self.tokensets.keys())
        tokensets_values = list(self.tokensets.values())
        dictionary = gensim.corpora.Dictionary(tokensets_values)
        dictionary.compactify()
        dictionary.save(self.current_workspace_path+"/Samples/"+self.key+'/ldadictionary.dict')
        logger.info("Dictionary created")
        raw_corpus = [dictionary.doc2bow(tokenset) for tokenset in tokensets_values]
        gensim.corpora.MmCorpus.serialize(self.current_workspace_path+"/Samples/"+self.key+'/ldacorpus.mm', raw_corpus)
        corpus = gensim.corpora.MmCorpus(self.current_workspace_path+"/Samples/"+self.key+'/ldacorpus.mm')
        logger.info("Corpus created")

        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = 'symmetric'
        if self.eta is not None:
            eta = self.eta
        else:
            eta = 'auto'
        
        cpus = psutil.cpu_count(logical=False)
        if cpus is None or cpus < 2:
            workers = 1
        else:
            workers = cpus-1

        model = gensim.models.ldamulticore.LdaMulticore(workers=workers,
                                                        corpus=corpus,
                                                        id2word=dictionary,
                                                        num_topics=self.num_topics,
                                                        passes=self.num_passes,
                                                        alpha=alpha,
                                                        eta=eta)
        model.save(self.current_workspace_path+"/Samples/"+self.key+'/ldamodel.lda', 'wb')
        logger.info("Completed generation of model")
        # Init output
        # capture all document topic probabilities both by document and by topic
        document_topic_prob = {}
        model_document_topics = model.get_document_topics(corpus, minimum_probability=0.0, minimum_phi_value=0)
        for doc_num in range(len(corpus)):
            doc_row = model_document_topics[doc_num]
            doc_topic_prob_row = {}
            for i, prob in doc_row:
                doc_topic_prob_row[i+1] = prob
            document_topic_prob[tokensets_keys[doc_num]] = doc_topic_prob_row

        logger.info("Finished")
        result={'key': self.key, 'document_topic_prob':document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))

class BitermTrainingThread(Thread):
    """BitermTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics, num_passes):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.num_passes = num_passes
        self.start()

    def run(self):
        '''Generates an Biterm model'''
        logger = logging.getLogger(__name__+"BitermTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        text_keys = []
        texts = []
        for key in self.tokensets:
            text_keys.append(key)
            text = ' '.join(self.tokensets[key])
            texts.append(text)

        logger.info("Starting generation of biterm model")

        X, vocab, vocab_dict = btm.get_words_freqs(texts)
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/vocab.pk', 'wb') as outfile:
            pickle.dump(vocab, outfile)
        logger.info("Vocab created")

        # Vectorizing documents
        docs_vec = btm.get_vectorized_docs(texts, vocab)
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/transformed_texts.pk', 'wb') as outfile:
            pickle.dump(docs_vec, outfile)
        logger.info("Texts transformed")

        logger.info("Starting Generation of BTM")
        biterms = btm.get_biterms(docs_vec)

        model = btm.BTM(X, vocab, T=self.num_topics, M=20, alpha=50/8, beta=0.01)
        p_zd = model.fit_transform(docs_vec, biterms, iterations=self.num_passes, verbose=False)
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/btm.pk', 'wb') as outfile:
            pickle.dump(model, outfile)
        logger.info("Completed Generation of BTM")

        document_topic_prob = {}
        for doc_num in range(len(p_zd)):
            doc_row = p_zd[doc_num]
            doc_topic_prob_row = {}
            for i in range(len(doc_row)):
                doc_topic_prob_row[i+1] = doc_row[i]
            document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

        logger.info("Finished")
        result={'key': self.key, 'document_topic_prob':document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))

class NMFTrainingThread(Thread):
    """NMFTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.start()

    def run(self):
        '''Generates an NMF model'''
        logger = logging.getLogger(__name__+"NMFTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        text_keys = []
        texts = []
        for key in self.tokensets:
            text_keys.append(key)
            text = ' '.join(self.tokensets[key])
            texts.append(text)

        logger.info("Starting generation of NMF model")

        tfidf_vectorizer = TfidfVectorizer(max_features=len(self.tokensets.values()), preprocessor=SamplesUtilities.dummy, tokenizer=SamplesUtilities.dummy, token_pattern=None)
        
        tfidf = tfidf_vectorizer.fit_transform(self.tokensets.values())
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
           pickle.dump(tfidf, outfile)
        
        logger.info("Texts transformed")

        logger.info("Starting Generation of NMF")

        model = nmf(self.num_topics, random_state=1).fit(tfidf)

        # must fit tfidf as above before saving tfidf_vectorizer
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
           pickle.dump(tfidf_vectorizer, outfile)

        topics = tfidf_vectorizer.get_feature_names_out()
        topic_pr = model.transform(tfidf)
        topic_pr_sum = numpy.sum(topic_pr, axis=1, keepdims=True)
        probs = numpy.divide(topic_pr, topic_pr_sum, out=numpy.zeros_like(topic_pr), where=topic_pr_sum!=0)

        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/nmf_model.pk', 'wb') as outfile:
            pickle.dump(model, outfile)
        logger.info("Completed Generation of NMF")

        document_topic_prob = {}
        for doc_num in range(len(probs)):
            doc_row = probs[doc_num]
            doc_topic_prob_row = {}
            for i in range(len(doc_row)):
                doc_topic_prob_row[i+1] = doc_row[i]
            document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

        logger.info("Finished")
        result={'key': self.key, 'document_topic_prob':document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


class Top2VecTrainingThread(Thread):
    """Top2VecTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.start()

    def run(self):
        '''Generates a Top2Vec model'''
        logger = logging.getLogger(__name__+"Top2VecTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        texts = [' '.join(tokenset) for tokenset in self.tokensets.values()]

        logger.info("Starting generation of Top2Vec model")

        model = Top2Vec(texts, embedding_model="distiluse-base-multilingual-cased", embedding_batch_size=2, min_count=10, workers=-1)

        logger.info("Top2Vec model generated")

        # Save the Top2Vec model
        model.save(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model.pk')
        logger.info("Top2Vec model saved")

        # Get document-topic probabilities
        topics, word_scores = model.get_topics()

        for i, topic in enumerate(topics):
            print(f"Topic {i + 1}: {', '.join(topic)}")
            
        tfidf_vectorizer = TfidfVectorizer(max_features=len(self.tokensets.values()), preprocessor=SamplesUtilities.dummy, tokenizer=SamplesUtilities.dummy, token_pattern=None)
        
        tfidf = tfidf_vectorizer.fit_transform(self.tokensets.values())
        
        # must fit tfidf as above before saving tfidf_vectorizer
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
           pickle.dump(tfidf_vectorizer, outfile)


        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model.pk', 'wb') as outfile:
            pickle.dump(model, outfile)
        logger.info("Completed Generation of Top2Vec")
        doc_ids_list = [str(doc_id) for doc_id in doc_ids_list]
        document_topic_prob = {}
        # Now you can pass the modified doc_ids_list to the get_documents_topics method
        document_topic_prob = model.get_documents_topics(doc_ids=doc_ids_list)

        # document_topic_prob = {}
        for doc_num in range(len(probs)):
            doc_row = probs[doc_num]
            doc_topic_prob_row = {}
            for i in range(len(doc_row)):
                doc_topic_prob_row[i+1] = doc_row[i]
            document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

        

        logger.info("Finished")
        result = {'key': self.key, 'document_topic_prob': document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


class BertopicTrainingThread(Thread):
    """BERTopicTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.start()

    def run(self):
        '''Generates a BERTopic model'''
        logger = logging.getLogger(__name__+"BertopicTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        text_keys = []
        texts = []
        for key in self.tokensets:
            text_keys.append(key)
            text = ' '.join(self.tokensets[key])
            texts.append(text)
        

        # if not torch.cuda.is_available():
        #      raise RuntimeError("CUDA is not available. Please ensure that your system has a compatible GPU and CUDA installed.")

        # Set device to CUDA
        device = torch.device("mps")
        # print(texts)
        # print(self.key)
        print("works!")
        logger.info("Starting generation of BERTopic model")
        
     
        # Step 2.1 - Extract embeddings
        # embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        print("works 1.1!")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_model = model.encode(texts, show_progress_bar=False)

        print("works 1.2!")
        
        # Step 2.2 - Reduce dimensionality
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        
        # Step 2.3 - Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        print("works 2!")
        # Step 2.4 - Tokenize topics
        vectorizer_model = CountVectorizer(stop_words="english")
        
        # Step 2.5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()
        
        topic_model = BERTopic(
            embedding_model=model,    # Step 1 - Extract embeddings
            umap_model=umap_model,              # Step 2 - Reduce dimensionality
            hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
            nr_topics=self.num_topics          # Step 6 - Diversify topic words
        )

        # Fit BERTopic model
        # ERROR IS HERE!!!!!!!!!! IT FAILS FROM THIS POINT!!!!!!! 
        # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # topic_model = BERTopic.load("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models_without_emb/pickle_without_emb/bertopic_17_03_2024", embedding_model=embedding_model )
        # pipe = make_pipeline(
        # TfidfVectorizer(),
        # TruncatedSVD(100)
        #     )
        # print("works 3!")
        # topic_model = BERTopic(embedding_model=pipe)
        
        # topics, probabilities = topic_model.fit_transform(texts)
        # print("Length of topics:", len(topics))
        # print("Length of probs:",len(probabilities))
        # # print("Shape of topics:",topics.shape)
        # # print("Shape of probabilities:",probabilities.shape)
        # logger.info("BERTopic model fitted")
        # pipe = make_pipeline(
        #     TfidfVectorizer(),
        #     TruncatedSVD(100)
        # )

        print("Pipeline created successfully")

        # topic_model = BERTopic(embedding_model=embedding_model)

        print("BERTopic model initialized successfully")

        topics, probabilities = topic_model.fit_transform(texts)
        # probabilities = probabilities.tolist()

        doc_info = topic_model.get_document_info(texts)
        doc_df = pd.DataFrame(doc_info)
        doc_info_csv_file = "doc_info_bertopic.csv"
        doc_df.to_csv(doc_info_csv_file, index=False)

        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
            pickle.dump(vectorizer_model, outfile)

        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
            pickle.dump(ctfidf_model, outfile)
        
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/bertopic_model.pk', 'wb') as outfile:
            pickle.dump(topic_model, outfile)


        print("BERTopic model fitted successfully")

        # print("Number of documents:", len(texts))
        # print("Number of topics:", len(topics))
        # print("Number of probabilities:", len(probabilities))
        # print(type(topics))
        # print(type(probabilities))

        # Print the first few elements of topics and probabilities to inspect their structure
        # print("First few elements of topics:", topics)
        # print("First few elements of probabilities:", probabilities)
        # Print the topic names associated with each topic ID
        # for topic_id in range(len(topic_model.get_topics())):
        #     topic_name = topic_model.get_topic(topic_id)
        #     print("Topic", topic_id, ":", topic_name)


        #worksss workssss
        document_topic_prob = {}
        print("works 5!")
        for doc_num, probs in enumerate(probabilities):
            doc_topic_prob_row = {}
            # print(doc_num)
            # print(probs)
            # print("works 5.5!")
            # Iterate over each probability in the current document
            for topic_id, prob in enumerate(probabilities):
                # print("works 5.6!")
                doc_topic_prob_row[topic_id] = prob
            # print("works 5.7!")
            document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
        
        result = {'key': self.key, 'document_topic_prob': document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
        print("works 7!")
       
            #worksss workssss




        #     document_topic_prob = {}
        #     for doc_num in range(len(probabilities)):
        #         doc_row = probabilities[doc_num]
        #         doc_topic_prob_row = {}
        #         for i in range(len(doc_row)):
        #             doc_topic_prob_row[i+1] = doc_row[i]
        #         document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
        #     result = {'key': self.key, 'document_topic_prob': document_topic_prob}
        #     wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
        #     print("works 7!")
        # except Exception as e:
        #     logger.error(f"Error occurred during BERTopic model creation: {e}")

            # Printing out the document_topic_prob to check
            # print(document_topic_prob)

            # for doc_num in range(len(probabilities)):
            #     print("Processing document number:", doc_num)

            #     # Retrieve the topic probabilities for the current document
            #     probs = probabilities[doc_num]
                

            #     # Print the length of the probabilities list for debugging
            #     print("Length of probabilities:", len(probs))
                

            #     # Iterate over the probabilities for each topic
            #     for topic_id, prob in enumerate(probs):
            #         # Print the probability for each topic for the current document
            #         print("Probability for topic", topic_id, ":", prob)

            #     # Retrieve the topic associated with the current document
            #     topic = topics[doc_num]
            #     print("Document topic:", topic)

            #     # Construct the document-topic probability row
            #     doc_topic_prob_row = {topic_id: prob for topic_id, prob in enumerate(probs)}

            #     # Store the document-topic probabilities in the dictionary
            #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row




            
            
            
            
            
            
            
            
            
            
            # print("works 4!")
            # #works till here
            # document_topic_prob = {}
            # for doc_num, (topic, prob) in enumerate(zip(topics, probabilities)):
            #     print("Processing document number:", doc_num)
            #     print("Document topic:", topic)
            #     print("Document length:", len(prob))
            #     doc_topic_prob_row = {topic: len(prob)}
            #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
            # result = {'key': self.key, 'document_topic_prob': document_topic_prob}


            
            # print("works 5!")
            #  #works till here
            # for doc_num in range(len(probabilities)):
            #     doc_topic_prob_row = {}
            #     print("works 5.5!")
            #      #works till here
            #     for topic_id, prob in enumerate(probabilities[doc_num]):
            #         doc_topic_prob_row[topic_id] = prob
            #         print("works 5.6!")
            #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
            # print("works 6!")
            # result = {'key': self.key, 'document_topic_prob': document_topic_prob}
            # wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# class Top2VecTrainingThread(Thread):
#     """Top2VecTrainingThread Class."""
#     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
#         """Init Worker Thread Class."""
#         Thread.__init__(self)
#         self.daemon = True
#         self._notify_window = notify_window
#         self.current_workspace_path = current_workspace_path
#         self.key = key
#         self.tokensets = tokensets
#         self.num_topics = num_topics
#         self.start()

#     def run(self):
#         '''Generates a Top2Vec model'''
#         logger = logging.getLogger(__name__+"Top2VecTrainingThread["+str(self.key)+"].run")
#         logger.info("Starting")
        
#         if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
#             os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

#         texts = [' '.join(tokenset) for tokenset in self.tokensets.values()]

#         logger.info("Starting generation of Top2Vec model")

#         model = Top2Vec(texts, embedding_model="distiluse-base-multilingual-cased", embedding_batch_size=2, workers = 4)

#         logger.info("Top2Vec model generated")

#         # Save the Top2Vec model
#         model.save(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model.pk')
#         logger.info("Top2Vec model saved")

#         # Get document-topic probabilities
#         document_topic_prob = model.get_documents_topics(doc_ids=self.tokensets.keys())

#         logger.info("Finished")
#         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
#         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# class BertopicTrainingThread(Thread):
#     """BERTopicTrainingThread Class."""
#     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
#         """Init Worker Thread Class."""
#         Thread.__init__(self)
#         self.daemon = True
#         self._notify_window = notify_window
#         self.current_workspace_path = current_workspace_path
#         self.key = key
#         self.tokensets = tokensets
#         self.num_topics = num_topics
#         self.start()

#     def run(self):
#         '''Generates a BERTopic model'''
#         logger = logging.getLogger(__name__ + "BertopicTrainingThread[" + str(self.key) + "].run")
#         logger.info("Starting")

#         if not os.path.exists(self.current_workspace_path + "/Samples/" + self.key):
#             os.makedirs(self.current_workspace_path + "/Samples/" + self.key)

#         text_keys = []
#         texts = []
#         for key in self.tokensets:
#             text_keys.append(key)
#             text = ' '.join(self.tokensets[key])
#             texts.append(text)

#         logger.info("Starting Loading of BERTopic model")

        
#             # loaded_model = BERTopic.load("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/pytorch", embedding_model=SentenceTransformer("all-MiniLM-L6-v2"), map_location=torch.device('cpu'))

#         topics = pickle.load(open("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/bertopic/topics_bertopic.pickle", "rb"))
#         probs = pickle.load(open("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/bertopic/probabilities_bertopic.pickle", "rb"))

#             # with bz2.BZ2File(self.current_workspace_path + "/Samples/" + self.key + '/bertopic_model.pk', 'wb') as outfile:
#             #     pickle.dump(loaded_model, outfile)

#         logger.info("Completed Generation of Bertopic")
#         # document_topic_prob = {}
#         # for doc_num in range(len(topics)):
#         #     doc_row = topics[doc_num]
#         #     doc_topic_prob_row = {}
#         #     for i in range(len(doc_row)):
#         #         doc_topic_prob_row[i+1] = doc_row[i]
#         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

#         document_topic_prob = {}
#         for doc_num in range(len(probs)):
#             if doc_num < len(text_keys):  # Check if doc_num is within the range of text_keys
#                 doc_row = probs[doc_num]
#                 logger.info(f"Processing document number {doc_num}")
#                 if isinstance(doc_row, np.float64):
#                     logger.warning("Encountered numpy float64, converting to list...")
#                     doc_row = [doc_row.item()]  # Convert numpy.float64 to a list with a single element
#                 logger.info(f"Length of doc_row: {len(doc_row)}")
#                 doc_topic_prob_row = {}
#                 for i in range(len(doc_row)):
#                     logger.info(f"Processing item number {i}")
#                     pass
#                 document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
#             else:
#                 logger.warning(f"doc_num {doc_num} exceeds the length of text_keys. Skipping...")

#         logger.info("Finished")
#         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
#         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
    