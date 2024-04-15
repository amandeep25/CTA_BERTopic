# # # # # def run(self):
# # # # #         '''Generates a BERTopic model'''
# # # # #         logger = logging.getLogger(__name__+"BertopicTrainingThread["+str(self.key)+"].run")
# # # # #         logger.info("Starting")
        
# # # # #         if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
# # # # #             os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

# # # # #         text_keys = []
# # # # #         texts = []
# # # # #         for key in self.tokensets:
# # # # #             text_keys.append(key)
# # # # #             text = ' '.join(self.tokensets[key])
# # # # #             texts.append(text)
    
# # # # #         # Set device to CUDA
# # # # #         device = torch.device("mps")
# # # # #         # print(texts)
# # # # #         # print(self.key)
# # # # #         print("works!")
# # # # #         logger.info("Starting generation of BERTopic model")
        
     
# # # # #         # Step 2.1 - Extract embeddings
# # # # #         print("works 1.1!")
# # # # #         model = SentenceTransformer("all-MiniLM-L6-v2")
# # # # #         embedding_model = model.encode(texts, show_progress_bar=False)

# # # # #         print("works 1.2!")
        
# # # # #         # Step 2.2 - Reduce dimensionality
# # # # #         umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        
# # # # #         # Step 2.3 - Cluster reduced embeddings
# # # # #         hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# # # # #         print("works 2!")
# # # # #         # Step 2.4 - Tokenize topics
# # # # #         vectorizer_model = CountVectorizer(stop_words="english")
        
# # # # #         # Step 2.5 - Create topic representation
# # # # #         ctfidf_model = ClassTfidfTransformer()
        
# # # # #         topic_model = BERTopic(
# # # # #             embedding_model=model,    # Step 1 - Extract embeddings
# # # # #             umap_model=umap_model,              # Step 2 - Reduce dimensionality
# # # # #             hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
# # # # #             vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
# # # # #             ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
# # # # #             nr_topics=self.num_topics          # Step 6 - Diversify topic words
# # # # #         )

   

# # # # #         topics, probabilities = topic_model.fit_transform(texts)
# # # # #         # probabilities = probabilities.tolist()


# # # # #         document_topic_prob = {}
# # # # #         print("works 5!")
# # # # #         for doc_num, probs in enumerate(probabilities):
# # # # #             doc_topic_prob_row = {}
# # # # #             # print(doc_num)
# # # # #             # print(probs)
# # # # #             # print("works 5.5!")
# # # # #             # Iterate over each probability in the current document
# # # # #             for topic_id, prob in enumerate(probabilities):
# # # # #                 # print("works 5.6!")
# # # # #                 doc_topic_prob_row[topic_id] = prob
# # # # #             # print("works 5.7!")
# # # # #             document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
        
# # # # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # # # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
# # # # #         print("works 7!")
       
# # # # #             #worksss workssss
        


# # # # #         for i in range(self.num_topics):
# # # # #             topic_num = i+1
# # # # #             self.parts_dict[topic_num] = BertopicTopicPart(self, topic_num, dataset)
# # # # #         self.parts_dict['unknown'] = TopicUnknownPart(self, 'unknown', [], dataset)

# # # # #         self.word_num = 10
# # # # #         self.ApplyDocumentCutoff()
        
# # # # #         self.end_dt = datetime.now()
# # # # #         logger.info("Finished")



# # # # #         def ApplyDocumentCutoff(self):
# # # # #         logger = logging.getLogger(__name__+"."+repr(self)+".ApplyDocumentCutoff")
# # # # #         logger.info("Starting")
# # # # #         document_set = set()
# # # # #         document_topic_prob_df = pd.DataFrame(data=self.document_topic_prob).transpose()

# # # # #         def UpdateLDATopicPart(topic):
# # # # #             document_list = []
# # # # #             document_s = document_topic_prob_df[topic].sort_values(ascending=False)
# # # # #             document_list = document_s.index[document_s >= self.document_cutoff].tolist()
# # # # #             document_set.update(document_list)
# # # # #             self.parts_dict[topic].part_data = document_list

# # # # #         for topic in self.parts_dict:
# # # # #             if isinstance(self.parts_dict[topic], Part) and topic != 'unknown':
# # # # #                 UpdateLDATopicPart(topic)
# # # # #             elif isinstance(self.parts_dict[topic], MergedPart):
# # # # #                 for subtopic in self.parts_dict[topic].parts_dict:
# # # # #                     if isinstance(self.parts_dict[topic].parts_dict[subtopic], Part) and topic != 'unknown':
# # # # #                         UpdateLDATopicPart(topic)


# # # # # import numpy as np

# # # # # # Your list
# # # # # my_list = [1, 2, 3, 4, 5]

# # # # # # Convert list to NumPy array
# # # # # my_array = np.array(my_list)

# # # # # # Now my_array is a NumPy ndarray
# # # # # print("Type of my_array:", type(my_array))
# # # # # print("Contents of my_array:", my_array)

# # # # import logging
# # # # # from threading import Thread
# # # # import os
# # # # import psutil
# # # # import bz2
# # # # import pickle
# # # # import numpy
# # # # import pickle
# # # # import torch
# # # # import numpy as np
# # # # import torch.multiprocessing as mp
# # # # import pandas as pd

# # # # import wx

# # # # #ML libraries
# # # # import gensim
# # # # #If not getting updated, use this code https://github.com/maximtrp/bitermplus
# # # # import bitermplus as btm
# # # # from sklearn.decomposition import NMF as nmf
# # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # from top2vec import Top2Vec
# # # # from bertopic import BERTopic
# # # # from sentence_transformers import SentenceTransformer
# # # # from umap import UMAP
# # # # from hdbscan import HDBSCAN
# # # # from nltk.stem import WordNetLemmatizer
# # # # from nltk.tokenize import word_tokenize
# # # # from sklearn.feature_extraction.text import CountVectorizer
# # # # from bertopic.vectorizers import ClassTfidfTransformer
# # # # from gensim.models.coherencemodel import CoherenceModel
# # # # from sklearn.pipeline import make_pipeline
# # # # from sklearn.decomposition import TruncatedSVD
# # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # from sentence_transformers import SentenceTransformer, util


# # # # from bertopic import BERTopic

# # # # # Example texts
# # # # texts = ["This is an example text for BERTopic.", "BERTopic is a topic modeling technique based on BERT embeddings."]
# # # # print("works 1.1!")
# # # # model = SentenceTransformer("all-MiniLM-L6-v2")
# # # # embedding_model = model.encode(texts, show_progress_bar=False)

# # # # print("works 1.2!")

# # # # # Step 2.2 - Reduce dimensionality
# # # # umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# # # # # Step 2.3 - Cluster reduced embeddings
# # # # hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# # # # print("works 2!")
# # # # # Step 2.4 - Tokenize topics
# # # # vectorizer_model = CountVectorizer(stop_words="english")

# # # # # Step 2.5 - Create topic representation
# # # # ctfidf_model = ClassTfidfTransformer()

# # # # topic_model = BERTopic(
# # # #     embedding_model=model,    # Step 1 - Extract embeddings
# # # #     umap_model=umap_model,              # Step 2 - Reduce dimensionality
# # # #     hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
# # # #     vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
# # # #     ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
# # # #     nr_topics=10        # Step 6 - Diversify topic words
# # # # )
# # # # # Initialize BERTopic model
# # # # # topic_model = BERTopic()

# # # # # Fit BERTopic model to texts
# # # # topics, probs = topic_model.fit_transform(texts)

# # # # # Retrieve topic words
# # # # topic_freq = topic_model.get_topic_freq()

# # # # # Display topic words
# # # # if 'Words' in topic_freq.columns:
# # # #     topic_names = topic_freq['Words'].values
# # # #     print("Topic Words:")
# # # #     print(topic_names)
# # # # else:
# # # #     print("Column 'Words' not found in topic frequency DataFrame.")




# # # # class BertopicTrainingThread(Thread):
# # # #     """BERTopicTrainingThread Class."""
# # # #     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
# # # #         """Init Worker Thread Class."""
# # # #         Thread.__init__(self)
# # # #         self.daemon = True
# # # #         self._notify_window = notify_window
# # # #         self.current_workspace_path = current_workspace_path
# # # #         self.key = key
# # # #         self.tokensets = tokensets
# # # #         self.num_topics = num_topics
# # # #         self.start()

# # # #     def run(self):
# # # #         '''Generates a BERTopic model'''
# # # #         logger = logging.getLogger(__name__+"BertopicTrainingThread["+str(self.key)+"].run")
# # # #         logger.info("Starting")
        
# # # #         if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
# # # #             os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

# # # #         text_keys = []
# # # #         texts = []
# # # #         for key in self.tokensets:
# # # #             text_keys.append(key)
# # # #             text = ' '.join(self.tokensets[key])
# # # #             texts.append(text)
    
# # # #         # Set device to CUDA
# # # #         device = torch.device("mps")
# # # #         # print(texts)
# # # #         # print(self.key)
# # # #         print("works!")
# # # #         logger.info("Starting generation of BERTopic model")
        
     
# # # #         # Step 2.1 - Extract embeddings
# # # #         print("works 1.1!")
# # # #         model = SentenceTransformer("all-MiniLM-L6-v2")
# # # #         embedding_model = model.encode(texts, show_progress_bar=False)

# # # #         print("works 1.2!")
        
# # # #         # Step 2.2 - Reduce dimensionality
# # # #         umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        
# # # #         # Step 2.3 - Cluster reduced embeddings
# # # #         hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# # # #         print("works 2!")
# # # #         # Step 2.4 - Tokenize topics
# # # #         vectorizer_model = CountVectorizer(stop_words="english")
        
# # # #         # Step 2.5 - Create topic representation
# # # #         ctfidf_model = ClassTfidfTransformer()
        
# # # #         topic_model = BERTopic(
# # # #             embedding_model=model,    # Step 1 - Extract embeddings
# # # #             umap_model=umap_model,              # Step 2 - Reduce dimensionality
# # # #             hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
# # # #             vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
# # # #             ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
# # # #             nr_topics=self.num_topics          # Step 6 - Diversify topic words
# # # #         )
# # # #         nr_topics=self.num_topics
# # # #         print("Pipeline created successfully")
# # # #         print("BERTopic model initialized successfully")

# # # #         # print(type(topics)) -> list originally
# # # #         # print(type(probs)) -> numpy ndarray 
# # # #         topics, probs = topic_model.fit_transform(texts)
# # # #         # freq = topic_model.get_topic_info()
# # # #         # print(freq.head(5))
# # # #         # Ensure that each probability array has 10 values
# # # #     #     probs_array = np.zeros((len(probs), self.num_topics))

# # # #     #     # Fill the array with probabilities
# # # #     #     for i, p in enumerate(probs):
# # # #     #         if isinstance(p, np.ndarray) and len(p) == self.num_topics:
# # # #     #             probs_array[i] = p

# # # #     #     # Assign probabilities
# # # #     #     probs = probs_array

# # # #     #     print("Topics:", topics)
# # # #     #     print("Probabilities:", probs)

# # # #     #    # Get the keywords for each topic
# # # #     #     topic_keywords = topic_model.get_topics()
# # # #     #     print("topics", topic_keywords)

# # # #     #     # Print the keywords for each topic
# # # #     #     for topic, keywords in topic_keywords.items():
# # # #     #         print(f"Topic {topic}:")
# # # #     #         for word, prob in keywords:
# # # #     #             print(f"- {word} (probability: {prob})")
# # # #     #         print()
                




# # # #         # # # Extract the topic names from the DataFrame
# # # #         # topic_names = freq['Representation'].values

# # # #         # # # Now you have a list of topic names corresponding to the topic numbers
# # # #         # print("Topic Names:", topic_names)

# # # #         # topics = np.array(topic_names)
# # # #         # probabilities = probabilities.tolist()
        

# # # #         doc_info = topic_model.get_document_info(texts)
# # # #         doc_df = pd.DataFrame(doc_info)
# # # #         doc_info_csv_file = "doc_info_bertopic.csv"
# # # #         doc_df.to_csv(doc_info_csv_file, index=False)

# # # #         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
# # # #             pickle.dump(vectorizer_model, outfile)

# # # #         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
# # # #             pickle.dump(ctfidf_model, outfile)
        
# # # #         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/bertopic_model.pk', 'wb') as outfile:
# # # #             pickle.dump(topic_model, outfile)


# # # #         print("BERTopic model fitted successfully")


# # # #         # document_topic_prob = {}
# # # #         # for doc_num in range(len(probs)):
# # # #         #     doc_row = probs[doc_num]
# # # #         #     doc_topic_prob_row = {}
# # # #         #     for i in range(len(doc_row)):
# # # #         #         doc_topic_prob_row[i+1] = doc_row[i]
# # # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row



# # # #         # document_topic_prob = {}
# # # #         # for doc_num in range(len(probs)):
# # # #         #     doc_topic_prob_row = {}
# # # #         #     # Ensure that probs[doc_num] is an iterable object (like a list or array)
# # # #         #     # if isinstance(probs[doc_num], (list, np.ndarray)):
# # # #         #         # Iterate over each topic and its corresponding probability
# # # #         #     for topic_id, prob in len(probs):
# # # #         #         doc_topic_prob_row[topic_id + 1] = prob
                   
# # # #         #         # Store the document's topic probabilities
# # # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row


# # # #         # # Example output
# # # #         # print(document_topic_prob)
# # # #         # document_topic_prob = {}
# # # #         # document_topic_prob = {}
      
# # # #         # for i, (topic, prob) in enumerate(zip(topics, probs)):
# # # #         #     doc_topic_prob_row = {str(t): p for t, p in zip(topic, prob)}  # Ensure topic is converted to string
# # # #         #     document_topic_prob[f'Document_{i+1}'] = doc_topic_prob_row

# # # #         # # Convert the document-topic probabilities dictionary to a DataFrame
# # # #         # df_document_topic_prob = pd.DataFrame(document_topic_prob).T
# # # #         # print(df_document_topic_prob.head(10))

# # # #         logger.info("Finished")

# # # #         # document_topic_prob = {}
# # # #         # for doc_num in range(len(probs)):
# # # #         #     doc_row = probs[doc_num]
# # # #         #     doc_topic_prob_row = {}
# # # #         #     for i in range(len(doc_row)):
# # # #         #         doc_topic_prob_row[i+1] = doc_row[i]
# # # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
       
    
# # # #         document_topic_prob = {}

# # # #         for doc_num in range(len(probs)):
# # # #             doc_topic_prob_row = {}

# # # #             # Ensure that probs[doc_num] is an iterable object (like a list or array)
# # # #             if isinstance(probs[doc_num], (list, np.ndarray)):
# # # #                 # Get the first probability value from probs[doc_num]
# # # #                 first_prob = probs[doc_num][0]

# # # #                 # Create an array of length num_topics and fill with zeros
# # # #                 doc_topic_prob_row = [0.0] * self.num_topics

# # # #                 # Assign the first probability value to the first position
# # # #                 doc_topic_prob_row[0] = first_prob
# # # #             else:
# # # #                 # Handle cases where probs[doc_num] is not iterable (e.g., if it's a single value)
# # # #                 first_prob = probs[doc_num]
                
# # # #                 # Create an array of length num_topics and fill with zeros
# # # #                 doc_topic_prob_row = [0.0] * self.num_topics
                
# # # #                 # Assign the first probability value to the first position
# # # #                 doc_topic_prob_row[0] = first_prob

# # # #             # Store the document's topic probabilities
# # # #             document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

# # # #         logger.info("Finished")

# # # #         # Create a result dictionary with the document-key and its associated document-topic probabilities
# # # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # # #         print(result)
# # # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# # # # #THIS works 
# # # #         # Step 4: Store document-topic probabilities
# # # #         # document_topic_prob = {}
# # # #         # for doc_num in range(len(probs)):
# # # #         #     doc_topic_prob_row = {}
# # # #         #     # Ensure that probs[doc_num] is an iterable object (like a list or array)
# # # #         #     if isinstance(probs[doc_num], (list, np.ndarray)):
# # # #         #         # Iterate over each topic and its corresponding probability
# # # #         #         for topic_id, prob in enumerate(probs[doc_num]):
# # # #         #             doc_topic_prob_row[topic_id + 1] = prob  # Increment topic_id by 1 to start from 1
# # # #         #         # Store the document's topic probabilities
# # # #         #         document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# # # #         #     else:
# # # #         #         # Handle cases where probs[doc_num] is not iterable (e.g., if it's a single value)
# # # #         #         doc_topic_prob_row[1] = probs[doc_num]  # Assuming 0 as the topic ID
# # # #         #         document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# # # #         # print(document_topic_prob)
        
# # # #         logger.info("Finished")
       

# # # # #         # Create a result dictionary with the document-key and its associated document-topic probabilities
# # # # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # # # #         print(result)
# # # # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
# # # # # #THIS works 


# # # #         #     document_topic_prob = {}
# # # #         #     for doc_num in range(len(probabilities)):
# # # #         #         doc_row = probabilities[doc_num]
# # # #         #         doc_topic_prob_row = {}
# # # #         #         for i in range(len(doc_row)):
# # # #         #             doc_topic_prob_row[i+1] = doc_row[i]
# # # #         #         document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# # # #         #     result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # # #         #     wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
# # # #         #     print("works 7!")
# # # #         # except Exception as e:
# # # #         #     logger.error(f"Error occurred during BERTopic model creation: {e}")

# # # #             # Printing out the document_topic_prob to check
# # # #             # print(document_topic_prob)

# # # #             # for doc_num in range(len(probabilities)):
# # # #             #     print("Processing document number:", doc_num)

# # # #             #     # Retrieve the topic probabilities for the current document
# # # #             #     probs = probabilities[doc_num]
                

# # # #             #     # Print the length of the probabilities list for debugging
# # # #             #     print("Length of probabilities:", len(probs))
                

# # # #             #     # Iterate over the probabilities for each topic
# # # #             #     for topic_id, prob in enumerate(probs):
# # # #             #         # Print the probability for each topic for the current document
# # # #             #         print("Probability for topic", topic_id, ":", prob)

# # # #             #     # Retrieve the topic associated with the current document
# # # #             #     topic = topics[doc_num]
# # # #             #     print("Document topic:", topic)

# # # #             #     # Construct the document-topic probability row
# # # #             #     doc_topic_prob_row = {topic_id: prob for topic_id, prob in enumerate(probs)}

# # # #             #     # Store the document-topic probabilities in the dictionary
# # # #             #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row


# # #       #this fails
# # #         # document_topic_prob = {}
# # #         # for i, (topic, prob) in enumerate(zip(topics, probs)):
# # #         #     doc_topic_prob_row = {str(t): p for t, p in enumerate(prob, start=1)}
# # #         #     document_topic_prob[text_keys[i]] = doc_topic_prob_row
# # #         # Fit BERTopic model on the text data
# # # #         topics, probs = topic_model.fit_transform(texts)
# # # #         print(topics)


            
            
            
            
            
            
            
            
            
            
# # # #             # print("works 4!")
# # # #             # #works till here
# # # #             # document_topic_prob = {}
# # # #             # for doc_num, (topic, prob) in enumerate(zip(topics, probabilities)):
# # # #             #     print("Processing document number:", doc_num)
# # # #             #     print("Document topic:", topic)
# # # #             #     print("Document length:", len(prob))
# # # #             #     doc_topic_prob_row = {topic: len(prob)}
# # # #             #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# # # #             # result = {'key': self.key, 'document_topic_prob': document_topic_prob}


            
# # # #             # print("works 5!")
# # # #             #  #works till here
# # # #             # for doc_num in range(len(probabilities)):
# # # #             #     doc_topic_prob_row = {}
# # # #             #     print("works 5.5!")
# # # #             #      #works till here
# # # #             #     for topic_id, prob in enumerate(probabilities[doc_num]):
# # # #             #         doc_topic_prob_row[topic_id] = prob
# # # #             #         print("works 5.6!")
# # # #             #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# # # #             # print("works 6!")
# # # #             # result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # # #             # wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# # # # # class Top2VecTrainingThread(Thread):
# # # # #     """Top2VecTrainingThread Class."""
# # # # #     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
# # # # #         """Init Worker Thread Class."""
# # # # #         Thread.__init__(self)
# # # # #         self.daemon = True
# # # # #         self._notify_window = notify_window
# # # # #         self.current_workspace_path = current_workspace_path
# # # # #         self.key = key
# # # # #         self.tokensets = tokensets
# # # # #         self.num_topics = num_topics
# # # # #         self.start()

# # # # #     def run(self):
# # # # #         '''Generates a Top2Vec model'''
# # # # #         logger = logging.getLogger(__name__+"Top2VecTrainingThread["+str(self.key)+"].run")
# # # # #         logger.info("Starting")
        
# # # # #         if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
# # # # #             os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

# # # # #         texts = [' '.join(tokenset) for tokenset in self.tokensets.values()]

# # # # #         logger.info("Starting generation of Top2Vec model")

# # # # #         model = Top2Vec(texts, embedding_model="distiluse-base-multilingual-cased", embedding_batch_size=2, workers = 4)

# # # # #         logger.info("Top2Vec model generated")

# # # # #         # Save the Top2Vec model
# # # # #         model.save(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model.pk')
# # # # #         logger.info("Top2Vec model saved")

# # # # #         # Get document-topic probabilities
# # # # #         document_topic_prob = model.get_documents_topics(doc_ids=self.tokensets.keys())

# # # # #         logger.info("Finished")
# # # # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # # # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# # # # # class BertopicTrainingThread(Thread):
# # # # #     """BERTopicTrainingThread Class."""
# # # # #     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
# # # # #         """Init Worker Thread Class."""
# # # # #         Thread.__init__(self)
# # # # #         self.daemon = True
# # # # #         self._notify_window = notify_window
# # # # #         self.current_workspace_path = current_workspace_path
# # # # #         self.key = key
# # # # #         self.tokensets = tokensets
# # # # #         self.num_topics = num_topics
# # # # #         self.start()

# # # # #     def run(self):
# # # # #         '''Generates a BERTopic model'''
# # # # #         logger = logging.getLogger(__name__ + "BertopicTrainingThread[" + str(self.key) + "].run")
# # # # #         logger.info("Starting")

# # # # #         if not os.path.exists(self.current_workspace_path + "/Samples/" + self.key):
# # # # #             os.makedirs(self.current_workspace_path + "/Samples/" + self.key)

# # # # #         text_keys = []
# # # # #         texts = []
# # # # #         for key in self.tokensets:
# # # # #             text_keys.append(key)
# # # # #             text = ' '.join(self.tokensets[key])
# # # # #             texts.append(text)

# # # # #         logger.info("Starting Loading of BERTopic model")

        
# # # # #             # loaded_model = BERTopic.load("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/pytorch", embedding_model=SentenceTransformer("all-MiniLM-L6-v2"), map_location=torch.device('cpu'))

# # # # #         topics = pickle.load(open("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/bertopic/topics_bertopic.pickle", "rb"))
# # # # #         probs = pickle.load(open("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/bertopic/probabilities_bertopic.pickle", "rb"))

# # # # #             # with bz2.BZ2File(self.current_workspace_path + "/Samples/" + self.key + '/bertopic_model.pk', 'wb') as outfile:
# # # # #             #     pickle.dump(loaded_model, outfile)

# # # # #         logger.info("Completed Generation of Bertopic")
# # # # #         # document_topic_prob = {}
# # # # #         # for doc_num in range(len(topics)):
# # # # #         #     doc_row = topics[doc_num]
# # # # #         #     doc_topic_prob_row = {}
# # # # #         #     for i in range(len(doc_row)):
# # # # #         #         doc_topic_prob_row[i+1] = doc_row[i]
# # # # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

# # # # #         document_topic_prob = {}
# # # # #         for doc_num in range(len(probs)):
# # # # #             if doc_num < len(text_keys):  # Check if doc_num is within the range of text_keys
# # # # #                 doc_row = probs[doc_num]
# # # # #                 logger.info(f"Processing document number {doc_num}")
# # # # #                 if isinstance(doc_row, np.float64):
# # # # #                     logger.warning("Encountered numpy float64, converting to list...")
# # # # #                     doc_row = [doc_row.item()]  # Convert numpy.float64 to a list with a single element
# # # # #                 logger.info(f"Length of doc_row: {len(doc_row)}")
# # # # #                 doc_topic_prob_row = {}
# # # # #                 for i in range(len(doc_row)):
# # # # #                     logger.info(f"Processing item number {i}")
# # # # #                     pass
# # # # #                 document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# # # # #             else:
# # # # #                 logger.warning(f"doc_num {doc_num} exceeds the length of text_keys. Skipping...")

# # # # #         logger.info("Finished")
# # # # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # # # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEve
        







# # # #         # Samples/ objects/ samples.py 
# # # #         class BertopicTopicPart(TopicPart):
# # # #     @property
# # # #     def word_num(self):
# # # #         return self._word_num
    
# # # #     # @word_num.setter
# # # #     # def word_num(self, value):
# # # #     #     logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].word_num")
# # # #     #     logger.info("Starting")

# # # #     #     if len(self.word_list) < value:
# # # #     #         self.word_list.clear()

# # # #     #         # Get the topic representation from the BERTopic model
# # # #     #         topic_representation = self.parent.model.get_topic(self.key)
# # # #     #         print(topic_representation)

# # # #     #         if isinstance(topic_representation, bool):
# # # #     #             logger.error("Error: Topic representation retrieval failed.")
# # # #     #         else:
# # # #     #             # Extract word probabilities for the specified topic
# # # #     #             word_probabilities = topic_representation[1]

# # # #     #             # Create a DataFrame to store word probabilities
# # # #     #             components_df = pd.DataFrame(word_probabilities, columns=['Probability'], index=topic_representation[0])

# # # #     #             components_df['Probability'] = pd.to_numeric(components_df['Probability'], errors='coerce')

# # # #     #             # Select top 'value' words with highest probabilities
# # # #     #             top_words = components_df.nlargest(value, 'Probability')

# # # #     #             # Extract word list and probability list
# # # #     #             word_list = top_words.index.tolist()
# # # #     #             prob_list = top_words['Probability'].tolist()

# # # #     #             # Combine word list and probability list into tuples
# # # #     #             word_prob_tuples = list(zip(word_list, prob_list))

# # # #     #             # Assign the combined list to 'self.word_list' attribute
# # # #     #             self.word_list = word_prob_tuples

# # # #     #     self._word_num = value
# # # #     #     logger.info("Finished")
    
# # # #     @word_num.setter
# # # #     def word_num(self, value):
# # # #         logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].word_num")
# # # #         logger.info("Starting")
        
# # # #         self._word_num = value
        
# # # #         if len(self.word_list) < value:
# # # #             self.word_list.clear()
# # # #             if isinstance(self.parent, ModelMergedPart):
                
# # # #             #     topic_words, word_probs = self.parent.parent.model.get_topic(self.key-1)

# # # #             #     # Convert the results into a list of tuples containing (word, probability)
# # # #             #     word_prob_tuples = list(zip(topic_words, word_probs))

# # # #             #     # Assign the list of tuples to self.word_list
# # # #             #     self.word_list = word_prob_tuples
                               
# # # #             # else:
# # # #             #     #This works 
# # # #             #     topic_words, word_probs = self.parent.model.get_topics()
# # # #             #     # print(topics)
                

# # # #             #     # Since get_topics returns only the words, initialize a list of zeros for their probabilities
# # # #             #     word_probs = [0.0] * len(topic_words)

# # # #             #     # Convert the results into a list of tuples containing (word, probability)
# # # #             #     word_prob_tuples = list(zip(topic_words, word_probs))

# # # #             #     # Assign the list of tuples to self.word_list
# # # #             #     self.word_list = word_prob_tuples
# # # # # #=======================================
# # # #                 # Assuming 'topic_model' is your BERTopic model object

# # # #                 # Get the topic representation from the BERTopic model
# # # #                 topic_representation = self.parent.model.get_topic(self.key)

# # # #                 # Extract word probabilities for the specified topic
# # # #                 word_probabilities = topic_representation[1]

# # # #                 # Create a DataFrame to store word probabilities
# # # #                 components_df = pd.DataFrame(word_probabilities, columns=['Probability'], index=topic_representation[0])

# # # #                 # Select top 'value' words with highest probabilities
# # # #                 top_words = components_df.nlargest(value, 'Probability')

# # # #                 # Extract word list and probability list
# # # #                 word_list = top_words.index.tolist()
# # # #                 prob_list = top_words['Probability'].tolist()

# # # #                 # Combine word list and probability list into tuples
# # # #                 word_prob_tuples = list(zip(word_list, prob_list))

# # # #                 # Assign the combined list to 'self.word_list' attribute
# # # #                 self.word_list = word_prob_tuples



# # # # # #=======================================
            
# # # #             # if self.key - 1 in topics:  # Check if the key exists in topics dictionary
# # # #             #     topic_words = topics[self.key - 1][:value]  # Assuming topics are 1-indexed
# # # #             #     print("topic words",  topic_words)
# # # #             #     self.word_list.extend(topic_words)
# # # #             #     print("word_list", self.word_list)

             
# # # #             # else:
# # # #             #     logger.warning("Topic key not found in topics dictionary.")
# # # #         self._word_num = value
# # # #         logger.info("Finished")
# # # #         self.last_changed_dt = datetime.now()
# # # #         logger.info("Finished")



# # # #         # # Get the topic information
# # # #         # topic_info = topic_model.get_topic_info()

# # # #         # # Extract the 'Name' column from the topic_info DataFrame
# # # #         # topics = topic_info['Name'].tolist()

# # # #         # topic_model.update_topics(texts, n_gram_range=(1,1))

# # # #         # # Update the topics in the BERTopic model
# # # #         # # topic_model.update_topic_model(texts, new_topics=new_topics)

# # # #         # # Print the updated topics
# # # #         # print(topic_model.get_topics())

# # # #         # # Get the document-topic probabilities
# # # #         # document_topic_prob = {}
# # # #         # for doc_num, (topic, prob) in enumerate(zip(topics, probs)):
# # # #         #     # Handle cases where topic is an integer (single topic per document)
# # # #         #     if isinstance(topic, int):
# # # #         #         doc_topic_prob_row = {str(i+1): 0.0 for i in range(self.num_topics)}
# # # #         #         doc_topic_prob_row[str(topic)] = prob
# # # #         #     # Handle cases where topic is a list of integers (multiple topics per document)
# # # #         #     elif isinstance(topic, (list, np.ndarray)):
# # # #         #         doc_topic_prob_row = {str(i+1): 0.0 for i in range(self.num_topics)}
# # # #         #         for t, p in zip(topic, prob):
# # # #         #             doc_topic_prob_row[str(t)] = p
# # # #         #     else:
# # # #         #         logger.error(f"Unexpected topic type: {type(topic)}")
# # # #         #         continue

# # # #         #     document_topic_prob[f"doc_{doc_num}"] = doc_topic_prob_row

# # # #         # result = {'key':  self.key, 'document_topic_prob': document_topic_prob}
# # # #         # print(result)
# # # #         # wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))




# # #     def ApplyDocumentCutoff(self):
# # #         logger = logging.getLogger(__name__+"."+repr(self)+".ApplyDocumentCutoff")
# # #         logger.info("Starting")
# # #         document_set = set()
# # #         document_topic_prob_df = pd.DataFrame(data=self.document_topic_prob).transpose()

# # #         print("DataFrame contents:")
# # #         print(document_topic_prob_df)

# # #         def UpdateLDATopicPart(topic):
# # #             logger = logging.getLogger(__name__ + ".UpdateLDATopicPart")
# # #             logger.info("Starting")

# # #             document_topic_prob_df = pd.DataFrame.from_dict(topic.document_topic_prob, orient='index')
# # #             document_s = document_topic_prob_df.sort_values(by=str(topic.key), ascending=False)
# # #             topic.word_num = 10
# # #             topic.word_list = document_s.head(topic.word_num).index.tolist()

# # #             logger.info("Finished")

# # #         for topic in self.parts_dict:
# # #             if isinstance(self.parts_dict[topic], Part) and topic != 'unknown':
# # #                 UpdateLDATopicPart(topic)
# # #             elif isinstance(self.parts_dict[topic], MergedPart):
# # #                 for subtopic in self.parts_dict[topic].parts_dict:
# # #                     if isinstance(self.parts_dict[topic].parts_dict[subtopic], Part) and topic != 'unknown':
# # #                         UpdateLDATopicPart(topic)
        
# # #         # Convert unknown_list set to a list
# # #         unknown_list = list(set(self._tokensets) - document_set)
# # #         unknown_df = document_topic_prob_df[document_topic_prob_df.index.isin(unknown_list)]
# # #         unknown_series = unknown_df.max(axis=1).sort_values()
# # #         new_unknown_list = list(unknown_series.index.values)

# # #         document_topic_prob_df["unknown"] = 0.0
# # #         document_topic_prob_df.loc[unknown_list, "unknown"] = 1.0
# # #         self.document_topic_prob = document_topic_prob_df.to_dict(orient='index')

# # #         self.parts_dict['unknown'].part_data = list(new_unknown_list)
# # #         logger.info("Finished")

# #      # Get the topic information
# #         # topic_info = topic_model.get_topics()

# #         # topic_info = topic_model.get_topic_info()

# #         # # Create a dictionary to map topic numbers to names
# #         # topic_names = {topic_id: topic_info.loc[topic_id, 'Name'] for topic_id in topic_info.index}
# #         # print("topic_names", topic_names)
# #         # print(len(topic_names))

# #         # # Use the topic_names dictionary to replace the topic numbers
# #         # topic_names_list = []
# #         # for topic in topics:
# #         #     if topic in topic_names:
# #         #         topic_names_list.append(topic_names[topic])
# #         #     else:
# #         #         topic_names_list.append(f"Topic {topic}")

# #         # print("topic_names_list", topic_names_list)
# #         # print(len(topic_names_list))
# #         # # topic_names_array = np.array(topic_names_list)




# #  # THIS WORKS ---------------------------
# #         # document_topic_prob = {}
      

# #       #this works gives different topic word as prob 
# #         # document_topic_prob = {}
# #         # for i, (topic, prob) in enumerate(zip(topics, probs)):
# #         #     doc_topic_prob_row = {str(topic): prob}
# #         #     document_topic_prob[text_keys[i]] = doc_topic_prob_row



# #     #    # Get the keywords for each topic
# #     #     topic_keywords = topic_model.get_topics()
# #     #     print("topics", topic_keywords)

# #     #     # Print the keywords for each topic
# #     #     for topic, keywords in topic_keywords.items():
# #     #         print(f"Topic {topic}:")
# #     #         for word, prob in keywords:
# #     #             print(f"- {word} (probability: {prob})")
# #     #         print()


# # # THIS WORKS ---------------------------

# # # class BertopicTrainingThread(Thread):
# # #     """BERTopicTrainingThread Class."""
# # #     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
# # #         """Init Worker Thread Class."""
# # #         Thread.__init__(self)
# # #         self.daemon = True
# # #         self._notify_window = notify_window
# # #         self.current_workspace_path = current_workspace_path
# # #         self.key = key
# # #         self.tokensets = tokensets
# # #         self.num_topics = num_topics
# # #         self.start()

# # #     def run(self):
# # #         '''Generates a BERTopic model'''
# # #         logger = logging.getLogger(__name__+"BertopicTrainingThread["+str(self.key)+"].run")
# # #         logger.info("Starting")
        
# # #         if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
# # #             os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

# # #         text_keys = []
# # #         texts = []
# # #         for key in self.tokensets:
# # #             text_keys.append(key)
# # #             text = ' '.join(self.tokensets[key])
# # #             texts.append(text)
    
# # #         # Set device to CUDA
# # #         device = torch.device("mps")
# # #         # print(texts)
# # #         # print(self.key)
# # #         print("works!")
# # #         logger.info("Starting generation of BERTopic model")
        
     
# # #         # Step 2.1 - Extract embeddings
# # #         print("works 1.1!")
# # #         model = SentenceTransformer("all-MiniLM-L6-v2")
# # #         embedding_model = model.encode(texts, show_progress_bar=False)

# # #         print("works 1.2!")
        
# # #         # Step 2.2 - Reduce dimensionality
# # #         umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        
# # #         # Step 2.3 - Cluster reduced embeddings
# # #         hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# # #         print("works 2!")
# # #         # Step 2.4 - Tokenize topics
# # #         vectorizer_model = CountVectorizer(stop_words="english")
        
# # #         # Step 2.5 - Create topic representation
# # #         ctfidf_model = ClassTfidfTransformer()
        
# # #         topic_model = BERTopic(
# # #             embedding_model=model,    # Step 1 - Extract embeddings
# # #             umap_model=umap_model,              # Step 2 - Reduce dimensionality
# # #             hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
# # #             vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
# # #             ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
# # #             nr_topics=self.num_topics          # Step 6 - Diversify topic words
# # #         )
# # #         nr_topics=self.num_topics
# # #         print("Pipeline created successfully")
# # #         print("BERTopic model initialized successfully")

# # #         # print(type(topics)) -> list originally
# # #         # print(type(probs)) -> numpy ndarray 
# # #         topics, probs = topic_model.fit_transform(texts)
# # #         # freq = topic_model.get_topic_info()
# # #         # print(freq.head(5))
# # #         # Ensure that each probability array has 10 values
# # #     #     probs_array = np.zeros((len(probs), self.num_topics))

# # #     #     # Fill the array with probabilities
# # #     #     for i, p in enumerate(probs):
# # #     #         if isinstance(p, np.ndarray) and len(p) == self.num_topics:
# # #     #             probs_array[i] = p

# # #     #     # Assign probabilities
# # #     #     probs = probs_array

# # #     #     print("Topics:", topics)
# # #     #     print("Probabilities:", probs)

# # #     #    # Get the keywords for each topic
# # #     #     topic_keywords = topic_model.get_topics()
# # #     #     print("topics", topic_keywords)

# # #     #     # Print the keywords for each topic
# # #     #     for topic, keywords in topic_keywords.items():
# # #     #         print(f"Topic {topic}:")
# # #     #         for word, prob in keywords:
# # #     #             print(f"- {word} (probability: {prob})")
# # #     #         print()
                




# # #         # # # Extract the topic names from the DataFrame
# # #         # topic_names = freq['Representation'].values

# # #         # # # Now you have a list of topic names corresponding to the topic numbers
# # #         # print("Topic Names:", topic_names)

# # #         # topics = np.array(topic_names)
# # #         # probabilities = probabilities.tolist()
        

# # #         doc_info = topic_model.get_document_info(texts)
# # #         doc_df = pd.DataFrame(doc_info)
# # #         doc_info_csv_file = "doc_info_bertopic.csv"
# # #         doc_df.to_csv(doc_info_csv_file, index=False)

# # #         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
# # #             pickle.dump(vectorizer_model, outfile)

# # #         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
# # #             pickle.dump(ctfidf_model, outfile)
        
# # #         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/bertopic_model.pk', 'wb') as outfile:
# # #             pickle.dump(topic_model, outfile)


# # #         print("BERTopic model fitted successfully")


# # #         # document_topic_prob = {}
# # #         # for doc_num in range(len(probs)):
# # #         #     doc_row = probs[doc_num]
# # #         #     doc_topic_prob_row = {}
# # #         #     for i in range(len(doc_row)):
# # #         #         doc_topic_prob_row[i+1] = doc_row[i]
# # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row



# # #         # document_topic_prob = {}
# # #         # for doc_num in range(len(probs)):
# # #         #     doc_topic_prob_row = {}
# # #         #     # Ensure that probs[doc_num] is an iterable object (like a list or array)
# # #         #     # if isinstance(probs[doc_num], (list, np.ndarray)):
# # #         #         # Iterate over each topic and its corresponding probability
# # #         #     for topic_id, prob in len(probs):
# # #         #         doc_topic_prob_row[topic_id + 1] = prob
                   
# # #         #         # Store the document's topic probabilities
# # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row


# # #         # # Example output
# # #         # print(document_topic_prob)
# # #         # document_topic_prob = {}
# # #         # document_topic_prob = {}
      
# # #         # for i, (topic, prob) in enumerate(zip(topics, probs)):
# # #         #     doc_topic_prob_row = {str(t): p for t, p in zip(topic, prob)}  # Ensure topic is converted to string
# # #         #     document_topic_prob[f'Document_{i+1}'] = doc_topic_prob_row

# # #         # # Convert the document-topic probabilities dictionary to a DataFrame
# # #         # df_document_topic_prob = pd.DataFrame(document_topic_prob).T
# # #         # print(df_document_topic_prob.head(10))

# # #         logger.info("Finished")

# # #         # document_topic_prob = {}
# # #         # for doc_num in range(len(probs)):
# # #         #     doc_row = probs[doc_num]
# # #         #     doc_topic_prob_row = {}
# # #         #     for i in range(len(doc_row)):
# # #         #         doc_topic_prob_row[i+1] = doc_row[i]
# # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
       
    
# # #         document_topic_prob = {}

# # #         for doc_num in range(len(probs)):
# # #             doc_topic_prob_row = {}

# # #             # Ensure that probs[doc_num] is an iterable object (like a list or array)
# # #             if isinstance(probs[doc_num], (list, np.ndarray)):
# # #                 # Get the first probability value from probs[doc_num]
# # #                 first_prob = probs[doc_num][0]

# # #                 # Create an array of length num_topics and fill with zeros
# # #                 doc_topic_prob_row = [0.0] * self.num_topics

# # #                 # Assign the first probability value to the first position
# # #                 doc_topic_prob_row[0] = first_prob
# # #             else:
# # #                 # Handle cases where probs[doc_num] is not iterable (e.g., if it's a single value)
# # #                 first_prob = probs[doc_num]
                
# # #                 # Create an array of length num_topics and fill with zeros
# # #                 doc_topic_prob_row = [0.0] * self.num_topics
                
# # #                 # Assign the first probability value to the first position
# # #                 doc_topic_prob_row[0] = first_prob

# # #             # Store the document's topic probabilities
# # #             document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

# # #         logger.info("Finished")

# # #         # Create a result dictionary with the document-key and its associated document-topic probabilities
# # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # #         print(result)
# # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# # #THIS works 
# #         # Step 4: Store document-topic probabilities
# #         # document_topic_prob = {}
# #         # for doc_num in range(len(probs)):
# #         #     doc_topic_prob_row = {}
# #         #     # Ensure that probs[doc_num] is an iterable object (like a list or array)
# #         #     if isinstance(probs[doc_num], (list, np.ndarray)):
# #         #         # Iterate over each topic and its corresponding probability
# #         #         for topic_id, prob in enumerate(probs[doc_num]):
# #         #             doc_topic_prob_row[topic_id + 1] = prob  # Increment topic_id by 1 to start from 1
# #         #         # Store the document's topic probabilities
# #         #         document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# #         #     else:
# #         #         # Handle cases where probs[doc_num] is not iterable (e.g., if it's a single value)
# #         #         doc_topic_prob_row[1] = probs[doc_num]  # Assuming 0 as the topic ID
# #         #         document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# #         # print(document_topic_prob)
        
# #         logger.info("Finished")
       

# # #         # Create a result dictionary with the document-key and its associated document-topic probabilities
# # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # #         print(result)
# # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
# # # #THIS works 


# #         #     document_topic_prob = {}
# #         #     for doc_num in range(len(probabilities)):
# #         #         doc_row = probabilities[doc_num]
# #         #         doc_topic_prob_row = {}
# #         #         for i in range(len(doc_row)):
# #         #             doc_topic_prob_row[i+1] = doc_row[i]
# #         #         document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# #         #     result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# #         #     wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
# #         #     print("works 7!")
# #         # except Exception as e:
# #         #     logger.error(f"Error occurred during BERTopic model creation: {e}")

# #             # Printing out the document_topic_prob to check
# #             # print(document_topic_prob)

# #             # for doc_num in range(len(probabilities)):
# #             #     print("Processing document number:", doc_num)

# #             #     # Retrieve the topic probabilities for the current document
# #             #     probs = probabilities[doc_num]
                

# #             #     # Print the length of the probabilities list for debugging
# #             #     print("Length of probabilities:", len(probs))
                

# #             #     # Iterate over the probabilities for each topic
# #             #     for topic_id, prob in enumerate(probs):
# #             #         # Print the probability for each topic for the current document
# #             #         print("Probability for topic", topic_id, ":", prob)

# #             #     # Retrieve the topic associated with the current document
# #             #     topic = topics[doc_num]
# #             #     print("Document topic:", topic)

# #             #     # Construct the document-topic probability row
# #             #     doc_topic_prob_row = {topic_id: prob for topic_id, prob in enumerate(probs)}

# #             #     # Store the document-topic probabilities in the dictionary
# #             #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row




            
            
            
            
            
            
            
            
            
            
# #             # print("works 4!")
# #             # #works till here
# #             # document_topic_prob = {}
# #             # for doc_num, (topic, prob) in enumerate(zip(topics, probabilities)):
# #             #     print("Processing document number:", doc_num)
# #             #     print("Document topic:", topic)
# #             #     print("Document length:", len(prob))
# #             #     doc_topic_prob_row = {topic: len(prob)}
# #             #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# #             # result = {'key': self.key, 'document_topic_prob': document_topic_prob}


            
# #             # print("works 5!")
# #             #  #works till here
# #             # for doc_num in range(len(probabilities)):
# #             #     doc_topic_prob_row = {}
# #             #     print("works 5.5!")
# #             #      #works till here
# #             #     for topic_id, prob in enumerate(probabilities[doc_num]):
# #             #         doc_topic_prob_row[topic_id] = prob
# #             #         print("works 5.6!")
# #             #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# #             # print("works 6!")
# #             # result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# #             # wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# # # class Top2VecTrainingThread(Thread):
# # #     """Top2VecTrainingThread Class."""
# # #     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
# # #         """Init Worker Thread Class."""
# # #         Thread.__init__(self)
# # #         self.daemon = True
# # #         self._notify_window = notify_window
# # #         self.current_workspace_path = current_workspace_path
# # #         self.key = key
# # #         self.tokensets = tokensets
# # #         self.num_topics = num_topics
# # #         self.start()

# # #     def run(self):
# # #         '''Generates a Top2Vec model'''
# # #         logger = logging.getLogger(__name__+"Top2VecTrainingThread["+str(self.key)+"].run")
# # #         logger.info("Starting")
        
# # #         if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
# # #             os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

# # #         texts = [' '.join(tokenset) for tokenset in self.tokensets.values()]

# # #         logger.info("Starting generation of Top2Vec model")

# # #         model = Top2Vec(texts, embedding_model="distiluse-base-multilingual-cased", embedding_batch_size=2, workers = 4)

# # #         logger.info("Top2Vec model generated")

# # #         # Save the Top2Vec model
# # #         model.save(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model.pk')
# # #         logger.info("Top2Vec model saved")

# # #         # Get document-topic probabilities
# # #         document_topic_prob = model.get_documents_topics(doc_ids=self.tokensets.keys())

# # #         logger.info("Finished")
# # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# # # class BertopicTrainingThread(Thread):
# # #     """BERTopicTrainingThread Class."""
# # #     def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
# # #         """Init Worker Thread Class."""
# # #         Thread.__init__(self)
# # #         self.daemon = True
# # #         self._notify_window = notify_window
# # #         self.current_workspace_path = current_workspace_path
# # #         self.key = key
# # #         self.tokensets = tokensets
# # #         self.num_topics = num_topics
# # #         self.start()

# # #     def run(self):
# # #         '''Generates a BERTopic model'''
# # #         logger = logging.getLogger(__name__ + "BertopicTrainingThread[" + str(self.key) + "].run")
# # #         logger.info("Starting")

# # #         if not os.path.exists(self.current_workspace_path + "/Samples/" + self.key):
# # #             os.makedirs(self.current_workspace_path + "/Samples/" + self.key)

# # #         text_keys = []
# # #         texts = []
# # #         for key in self.tokensets:
# # #             text_keys.append(key)
# # #             text = ' '.join(self.tokensets[key])
# # #             texts.append(text)

# # #         logger.info("Starting Loading of BERTopic model")

        
# # #             # loaded_model = BERTopic.load("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/pytorch", embedding_model=SentenceTransformer("all-MiniLM-L6-v2"), map_location=torch.device('cpu'))

# # #         topics = pickle.load(open("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/bertopic/topics_bertopic.pickle", "rb"))
# # #         probs = pickle.load(open("/Users/a3kaur/Documents/python_venvs/ctaenv/src/CTA_new-main-2/src/Saved_models/bertopic/probabilities_bertopic.pickle", "rb"))

# # #             # with bz2.BZ2File(self.current_workspace_path + "/Samples/" + self.key + '/bertopic_model.pk', 'wb') as outfile:
# # #             #     pickle.dump(loaded_model, outfile)

# # #         logger.info("Completed Generation of Bertopic")
# # #         # document_topic_prob = {}
# # #         # for doc_num in range(len(topics)):
# # #         #     doc_row = topics[doc_num]
# # #         #     doc_topic_prob_row = {}
# # #         #     for i in range(len(doc_row)):
# # #         #         doc_topic_prob_row[i+1] = doc_row[i]
# # #         #     document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

# # #         document_topic_prob = {}
# # #         for doc_num in range(len(probs)):
# # #             if doc_num < len(text_keys):  # Check if doc_num is within the range of text_keys
# # #                 doc_row = probs[doc_num]
# # #                 logger.info(f"Processing document number {doc_num}")
# # #                 if isinstance(doc_row, np.float64):
# # #                     logger.warning("Encountered numpy float64, converting to list...")
# # #                     doc_row = [doc_row.item()]  # Convert numpy.float64 to a list with a single element
# # #                 logger.info(f"Length of doc_row: {len(doc_row)}")
# # #                 doc_topic_prob_row = {}
# # #                 for i in range(len(doc_row)):
# # #                     logger.info(f"Processing item number {i}")
# # #                     pass
# # #                 document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
# # #             else:
# # #                 logger.warning(f"doc_num {doc_num} exceeds the length of text_keys. Skipping...")

# # #         logger.info("Finished")
# # #         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
# # #         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


# # def ApplyDocumentCutoff(self):
#     #     logger = logging.getLogger(__name__+"."+repr(self)+".ApplyDocumentCutoff")
#     #     logger.info("Starting")
#     #     document_set = set()
#     #     # document_topic_prob_df = pd.DataFrame(data=self.document_topic_prob)
#     #     document_topic_prob_df = pd.DataFrame(data=self.document_topic_prob).transpose()
#     #     # document_topic_prob_df = document_topic_prob_df.drop(columns=['0'])

#     #     print("DataFrame contents:")
#     #     print(document_topic_prob_df)

#     #     def UpdateLDATopicPart(topic):
#     #         document_topic_prob_df = pd.DataFrame.from_dict(topic.document_topic_prob, orient='index')
#     #         document_s = document_topic_prob_df.sort_values(by=str(topic.key), ascending=False)
#     #         topic.word_num = 10
#     #         topic.word_list = document_s.head(topic.word_num).index.tolist()
#     #         document_list = []
#     #         # print(document_topic_prob_df.head())
           
#     #         document_list = document_s.index[document_s >= self.document_cutoff].tolist()
#     #         document_set.update(document_list)
#     #         self.parts_dict[topic].part_data = document_list

#     #     for topic in self.parts_dict:
#     #         if isinstance(self.parts_dict[topic], Part) and topic != 'unknown':
#     #             UpdateLDATopicPart(topic)
#     #         elif isinstance(self.parts_dict[topic], MergedPart):
#     #             for subtopic in self.parts_dict[topic].parts_dict:
#     #                 if isinstance(self.parts_dict[topic].parts_dict[subtopic], Part) and topic != 'unknown':
#     #                     UpdateLDATopicPart(topic)
        
#     #     # Convert unknown_list set to a list
#     #     unknown_list = list(set(self._tokensets) - document_set)
#     #     unknown_df = document_topic_prob_df[document_topic_prob_df.index.isin(unknown_list)]
#     #     unknown_series = unknown_df.max(axis=1).sort_values()
#     #     new_unknown_list = list(unknown_series.index.values)

#     #     document_topic_prob_df["unknown"] = 0.0
#     #     document_topic_prob_df.loc[unknown_list, "unknown"] = 1.0
#     #     self.document_topic_prob = document_topic_prob_df.to_dict(orient='index')

#     #     self.parts_dict['unknown'].part_data = list(new_unknown_list)
#     #     logger.info("Finished")

#     # def ApplyDocumentCutoff(self):
#     #     logger = logging.getLogger(__name__ + "." + repr(self) + ".ApplyDocumentCutoff")
#     #     logger.info("Starting")
        
#     #     document_set = set()
#     #     document_topic_prob_df = pd.DataFrame(data=self.document_topic_prob).transpose()

#     #     print("DataFrame contents:")
#     #     print(document_topic_prob_df)

#     #     # unknown_list = ['Reddit', 'discussion', '11y57hg']  # Example list, replace with actual list

#     #     # document_topic_prob_df.loc[unknown_list, "unknown"] = 1.0

#     #     logger.info("Finished")

#     #     def UpdateLDATopicPart(topic):
#     #         document_list = []
#     #         try:
#     #             document_s = document_topic_prob_df[topic].sort_values(ascending=False)
#     #             document_list = document_s.index[document_s >= self.document_cutoff].tolist()
#     #             document_set.update(document_list)
#     #             self.parts_dict[topic].part_data = document_list
#     #         except KeyError as e:
#     #             logger.error(f"KeyError: {e}")
#     #             logger.error(f"Failed to update LDATopicPart for topic: {topic}")

#     #     for topic in self.parts_dict:
#     #         if isinstance(self.parts_dict[topic], Part) and topic != 'unknown':
#     #             UpdateLDATopicPart(topic)
#     #         elif isinstance(self.parts_dict[topic], MergedPart):
#     #             for subtopic in self.parts_dict[topic].parts_dict:
#     #                 if isinstance(self.parts_dict[topic].parts_dict[subtopic], Part) and topic != 'unknown':
#     #                     UpdateLDATopicPart(topic)

#     #     logger.info("Finished")
#     #     # unknown_list = set(self._tokensets) - document_set
#     #     # unknown_df = document_topic_prob_df[document_topic_prob_df.index.isin(unknown_list)]
#     #     # unknown_series = unknown_df.max(axis=1).sort_values()
#     #     # new_unknown_list = list(unknown_series.index.values)
        
#     #     # document_topic_prob_df["unknown"] = 0.0
#     #     # document_topic_prob_df.loc[unknown_list, "unknown"] = 1.0
#     #     # self.document_topic_prob = document_topic_prob_df.to_dict(orient='index')
                        
#     #     # Convert unknown_list set to a list
#     #     unknown_list = list(set(self._tokensets) - document_set)
#     #     existing_unknown_list = list(set(unknown_list) & set(document_topic_prob_df.index))  # Filter out non-existent index labels
#     #     unknown_df = document_topic_prob_df[document_topic_prob_df.index.isin(existing_unknown_list)]
#     #     unknown_series = unknown_df.max(axis=1).sort_values()
#     #     new_unknown_list = list(unknown_series.index.values)

#     #     document_topic_prob_df["unknown"] = 0.0
#     #     document_topic_prob_df.loc[new_unknown_list, "unknown"] = 1.0  # Use new_unknown_list
#     #     self.document_topic_prob = document_topic_prob_df.to_dict(orient='index')

#     #     self.parts_dict['unknown'].part_data = list(new_unknown_list)
#     #     logger.info("Finished")


# class BertopicTopicPart(TopicPart):
#         '''Instances of BERTopic objects'''
#         def __init__(self, parent, key, dataset, name=None):
#             if name is None:
#                 name = "BERTopic " + str(key)
#             ModelPart.__init__(self, parent, key, [], dataset, name)

#             # Properties that automatically update last_changed_dt
#             self._word_num = 0
#             self._word_list = []

#         def __repr__(self):
#             return 'BERTopic %s' % (self.key,) if self.label == "" else 'BERTopic %s: %s' % (self.key, self.label,)

#         @property
#         def word_num(self):
#             return self._word_num

#         @word_num.setter
#         def word_num(self, value):
#             self._word_num = value
#             self.last_changed_dt = datetime.now()

#         @property
#         def word_list(self):
#             return self._word_list

#         @word_list.setter
#         def word_list(self, value):
#             self._word_list = value
#             self.last_changed_dt = datetime.now()

#         def update_word_list(self):
#             logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].word_num")
#             logger.info("Starting")

#             if isinstance(self.parent, ModelMergedPart):
#                 topic_info = self.parent.parent.model.get_topic_info(self.key)
#             else:
#                 topic_info = self.parent.model.get_topic_info(self.key)

#             # Get the top 'self.word_num' words and their representation scores
#             top_words_and_scores = topic_info.nlargest(self.word_num, 'Representation')[['Word', 'Representation']].values.tolist()

#             # Get the documents associated with the current topic
#             topic_documents = self.get_topic_documents(self.key)

#             # Assign the top words, scores, and documents to the word_list
#             self.word_list = [(word, score, doc) for word, score in top_words_and_scores for doc in topic_documents]

#             self.last_changed_dt = datetime.now()
#             logger.info("Finished")

#         def get_topic_documents(self, topic_id):
#             logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].get_topic_documents")
#             logger.info("Starting")

#             topic_info = self.parent.model.get_topic_info(topic_id)
#             topic_documents = topic_info['Representative_Docs']

#             logger.info("Finished")
#             return topic_documents
#         # @property
      
#         # def word_num(self):
#         #     return self._word_num

#         # @word_num.setter
#         # def word_num(self, value):
#         #     self._word_num = value
#         #     self.last_changed_dt = datetime.now()

#         # @property
#         # def word_list(self):
#         #     return self._word_list

#         # @word_list.setter
#         # def word_list(self, value):
#         #     self._word_list = value
#         #     self.last_changed_dt = datetime.now()

#         # def update_word_list(self):
#         #     logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].word_num")
#         #     logger.info("Starting")

#         #     if isinstance(self.parent, ModelMergedPart):
#         #         topic_info = self.parent.parent.model.get_topic_info(self.key)
#         #         # print(topic_info)
#         #     else:
#         #         topic_info = self.parent.model.get_topic_info(self.key)
#         #         # print(topic_info)

#         #     # Get the top 'self.word_num' words and their representation scores
#         #     top_words_and_scores = topic_info.nlargest(self.word_num, 'Representation')[['Word', 'Representation']].values.tolist()
#         #     # print(top_words_and_scores)
#         #     # Assign the top words and scores to the word_list
#         #     self.word_list = top_words_and_scores

#         #     self.last_changed_dt = datetime.now()
#         #     logger.info("Finished")
    
#         # @word_num.setter
#         # def word_num(self, value):
#         #     logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].word_num")
#         #     logger.info("Starting")
#         #     self._word_num = value

#         #     if len(self.word_list) < value:
#         #         self.word_list.clear()

#         #         if isinstance(self.parent, ModelMergedPart):
#         #             topic_info = self.parent.parent.model.get_topic_info(self.key - 1)
#         #         else:
#         #             topic_info = self.parent.model.get_topic_info(self.key - 1)

#         #         # Get the top 'value' words and their representation scores
#         #         top_words_and_scores = topic_info.nlargest(value, 'Representation')[['Word', 'Representation']].values.tolist()

#         #         # Assign the top words and scores to the word_list
#         #         self.word_list = top_words_and_scores

#         #     self.last_changed_dt = datetime.now()
#         #     logger.info("Finished")



# # class BertopicTopicPart(TopicPart):
# #     @property
# #     def word_num(self):
# #         return self._word_num
    
    
# #     @word_num.setter
# #     def word_num(self, value):
# #         logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].word_num")
# #         logger.info("Starting")
        
# #         self._word_num = value
        
# #         if len(self.word_list) < value:
# #             self.word_list.clear()
# #             if isinstance(self.parent, ModelMergedPart):
                
# #                 topic_words, word_probs = self.parent.parent.model.get_topic(self.key-1)

# #                 # Convert the results into a list of tuples containing (word, probability)
# #                 word_prob_tuples = list(zip(topic_words, word_probs))

# #                 # Assign the list of tuples to self.word_list
# #                 self.word_list = word_prob_tuples
                               
# #             else:
# #                 #This works 
# #                 topic_words, word_probs = self.parent.model.get_topics()
# #                 # print(topics)
                

# #                 # Since get_topics returns only the words, initialize a list of zeros for their probabilities
# #                 word_probs = [0.0] * len(topic_words)

# #                 # Convert the results into a list of tuples containing (word, probability)
# #                 word_prob_tuples = list(zip(topic_words, word_probs))

# #                 # Assign the list of tuples to self.word_list
# #                 self.word_list = word_prob_tuples
        
    
# #         self._word_num = value
# #         logger.info("Finished")
# #         self.last_changed_dt = datetime.now()
# #         logger.info("Finished")
        
             
#         # document_topic_prob = np.hstack((document_topic_prob, np.zeros((document_topic_prob.shape[0], 1))))

#         # Convert the NumPy array to a DataFrame
#         # document_topic_prob_df = pd.DataFrame(document_topic_prob, columns=[f"Topic {i}" for i in range(1,document_topic_prob.shape[1])])
#         # # Fill NaN values with 0.0
#         # document_topic_prob_df.fillna(0.0, inplace=True)                        
        

#         # result = {'key': self.key, 'document_topic_prob': {}}

#         # for i, row in enumerate(document_topic_prob_df.values):
#         #     formatted_key = (text_keys[i],)  # Assuming text_keys contains the keys corresponding to the rows
#         #     result['document_topic_prob'][formatted_key] = {j + 1: prob for j, prob in enumerate(row)}

#         # # Construct final result
#         # # final_result = {'key': result['key'], 'document_topic_prob': result['document_topic_prob']}
#         # logger.info("Finished")
#         # wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))

#                 # topics, probs = topic_model.fit_transform(texts)

# # # # this works 
# #         document_topic_prob = {}
# #         for doc_num, (topic, prob) in enumerate(zip(topics, probs)):
# #             # Handle cases where topic is an integer (single topic per document)
# #             if isinstance(topic, int):
# #                 print("this works 1")
# #                 doc_topic_prob_row = {str(i+1): 0.0 for i in range(self.num_topics)}
# #                 doc_topic_prob_row[str(topic)] = prob
# #             # Handle cases where topic is a list of integers (multiple topics per document)
# #             elif isinstance(topic, (list, np.ndarray)):
# #                 print("this works 2")
# #                 doc_topic_prob_row = {str(i+1): 0.0 for i in range(self.num_topics)}
# #                 for t, p in zip(topic, prob):
# #                     doc_topic_prob_row[str(t+1)] = p
# #             else:
# #                 logger.error(f"Unexpected topic type: {type(topic)}")
# #                 continue

# #             document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row
        
#         # def word_num(self, value):
#         #     logger = logging.getLogger(__name__ + ".BERTopicPart[" + str(self.key) + "].word_num")
#         #     logger.info("Starting")
#         #     self._word_num = value
#         #     if len(self.word_list) < value:
#         #         self.word_list.clear()
#         #         if isinstance(self.parent, ModelMergedPart):
#         #             topic_words = self.parent.parent.model.get_topics(self.key)
#         #         else:
#         #             topic_words = self.parent.model.get_topics(self.key)
#         #             print(topic_words)
#         #         self.word_list.extend(topic_words)
#         #     self.last_changed_dt = datetime.now()
#         #     logger.info("Finished") 


# # class BertopicTopicPart(TopicPart):
# #     @property
# #     def word_num(self):
# #         return self._word_num

# #     @word_num.setter
# #     def word_num(self, value):
# #         logger = logging.getLogger(__name__ + ".BertopicTopicPart[" + str(self.key) + "].word_num")
# #         logger.info("Starting")

# #         if isinstance(self.parent, ModelMergedPart):
# #             topic_model = self.parent.parent.model
# #         else:
# #             topic_model = self.parent.model

# #         topic_words_dict = topic_model.get_topics()

# #         topic_word_lists = []
# #         for topic_id, word_prob_data in topic_words_dict.items():
# #             if isinstance(word_prob_data, dict):
# #                 # Sort the word probabilities in descending order
# #                 sorted_word_probs = sorted(word_prob_data.items(), key=lambda x: x[1], reverse=True)

# #                 # Get the top `word_num` words and probabilities
# #                 word_prob_list = [(word, prob) for word, prob in sorted_word_probs[:value]]
# #             elif isinstance(word_prob_data, list):
# #                 # If word_prob_data is a list, sort it by the second element (probability) in descending order
# #                 sorted_word_probs = sorted(word_prob_data, key=lambda x: x[1], reverse=True)

# #                 # Get the top `word_num` words and probabilities
# #                 word_prob_list = sorted_word_probs[:value]
# #             else:
# #                 logger.warning(f"Unexpected data type for topic {topic_id}: {type(word_prob_data)}")
# #                 continue

# #             topic_word_lists.append((topic_id, word_prob_list))

# #             self.word_list = topic_word_lists
# #             print(self.word_list)
# #             print("type:", type(self.word_list))
# #         self._word_num = value
#     #     logger.info("Finished")

#     #     class BertopicTopicPart(TopicPart):
#     # @property
#     # def word_num(self):
#     #     return self._word_num

#     # @word_num.setter
#     # def word_num(self, value):
#     #     logger = logging.getLogger(__name__ + ".BertopicTopicPart[" + str(self.key) + "].word_num")
#     #     logger.info("Starting")

#     #     if isinstance(self.parent, ModelMergedPart):
#     #         topic_model = self.parent.parent.model
#     #     else:
#     #         topic_model = self.parent.model

#     #     topic_words_dict = topic_model.get_topics()

#     #     topic_word_lists = []
#     #     for word_prob_list in topic_words_dict.values():
#     #         try:
#     #             # Sort the word probabilities in descending order
#     #             sorted_word_probs = sorted(word_prob_list, key=lambda x: x[1], reverse=True)
                
#     #             # Get the top `word_num` words and probabilities
#     #             top_words = [word_prob[0] for word_prob in sorted_word_probs[:self._word_num]]
#     #             top_probs = [word_prob[1] for word_prob in sorted_word_probs[:self._word_num]]

#     #             # Convert the list of tuples to a dictionary
#     #             word_prob_dict = dict(zip(top_words, top_probs))

#     #             # Append the dictionary to the result list
#     #             topic_word_lists.append(word_prob_dict)
                
#     #         except ValueError:
#     #             # Handle the ValueError gracefully
#     #             logger.warning("Empty word probability list encountered, skipping...")
#     #             continue

#     #     # Assign the list of dictionaries to the word_list property
#     #     self.word_list = topic_word_lists

#     #     # Set the word_num value
#     #     self._word_num = value
#     #     logger.info("Finished")
        
#         class Top2VecTrainingThread(Thread):
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

#         text_keys = []
#         texts = []
#         for key in self.tokensets:
#             text_keys.append(key)
#             text = ' '.join(self.tokensets[key])
#             texts.append(text)

#         logger.info("Starting generation of Top2Vec model")

#         model = Top2Vec(texts, embedding_model="distiluse-base-multilingual-cased", embedding_batch_size=32, min_count=10, workers=-1)

#         logger.info("Top2Vec model generated")

#         # Save the Top2Vec model
#         model.save(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model.pk')
#         logger.info("Top2Vec model saved")
#         tfidf_vectorizer = TfidfVectorizer(max_features=len(self.tokensets.values()), preprocessor=SamplesUtilities.dummy, tokenizer=SamplesUtilities.dummy, token_pattern=None)
        
#         tfidf = tfidf_vectorizer.fit_transform(self.tokensets.values())
        
#         # must fit tfidf as above before saving tfidf_vectorizer
#         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
#            pickle.dump(tfidf_vectorizer, outfile)


#         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model.pk', 'wb') as outfile:
#             pickle.dump(model, outfile)

#         kmeans = KMeans(n_clusters=self.num_topics, random_state=0).fit(model._get_document_embeddings())
#         labels = kmeans.labels_

#         document_topic_prob = {}
#         for doc_key, label in zip(self.tokensets.keys(), labels):
#             doc_topic_prob_row = {i: 0.0 for i in range(self.num_topics)}
#             doc_topic_prob_row[label] = 1.0
#             document_topic_prob[doc_key] = doc_topic_prob_row

#         logger.info("Finished")
#         result = {'key': self.key, 'document_topic_prob': document_topic_prob}
#         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))



#    def _find_topic_words_and_scores(self, topic_vectors):
#         topic_words = []
#         topic_word_scores = []

#         res = np.inner(topic_vectors, self.word_vectors)
#         top_words = np.flip(np.argsort(res, axis=1), axis=1)
#         top_scores = np.flip(np.sort(res, axis=1), axis=1)

#         for words, scores in zip(top_words, top_scores):
#             topic_words.append([self.vocab[i] for i in words[0:50]])
#             topic_word_scores.append(scores[0:50])

#         topic_words = np.array(topic_words)
#         topic_word_scores = np.array(topic_word_scores)

#         return topic_words, topic_word_scores


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
#         logger = logging.getLogger(__name__ + ".Top2VecTrainingThread[" + str(self.key) + "].run")
#         logger.info("Starting")

#         if not os.path.exists(self.current_workspace_path + "/Samples/" + self.key):
#             os.makedirs(self.current_workspace_path + "/Samples/" + self.key)

#         text_keys = []
#         texts = []
#         for key in self.tokensets:
#             text_keys.append(key)
#             text = ' '.join(self.tokensets[key])
#             texts.append(text)

#         logger.info("Starting generation of Top2Vec model")

#         model = Top2Vec(texts, embedding_model="distiluse-base-multilingual-cased", embedding_batch_size=32, min_count=10, workers=-1)
        
#         logger.info("Top2Vec model generated")

#         logger.info("Top2Vec model saved")
        
#         tfidf_vectorizer = TfidfVectorizer(max_features=len(self.tokensets.values()))
#         tfidf = tfidf_vectorizer.fit_transform(texts)
#         with bz2.BZ2File(self.current_workspace_path + "/Samples/" + self.key + '/tfidf_vectorizer.pk', 'wb') as outfile:
#             pickle.dump(tfidf_vectorizer, outfile)
        
#         with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
#             pickle.dump(tfidf, outfile)

#         # Save the Top2Vec model again
#         with bz2.BZ2File(self.current_workspace_path + "/Samples/" + self.key + '/top2vec_model.pk', 'wb') as outfile:
#             pickle.dump(model, outfile)


      

#         topics, word_scores, _ = model.get_topics()

 
#         print("len topics", len(topics))
#         print("len words-scores", len(word_scores))
#         result = {'key': self.key, 'document_topic_prob': {}}
#         for i, doc_key in enumerate(text_keys):
#             word_score = word_scores[i]
#             result['document_topic_prob'][doc_key] = {j + 1: prob for j, prob in enumerate(word_score[:20])}

#         logger.info("Finished")
#         doc_topic_probs = {text_keys[i]: {j + 1: prob for j, prob in enumerate(word_score[:20])} 
#                            for i, word_score in enumerate(word_scores)}
#         result = {'key': self.key, 'document_topic_prob': doc_topic_probs}
#         print(result)
#         wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))
#         logger.info("Completed Generation of Top2Vec")

# class Top2VecTopicPart(TopicPart):
#     @property
#     def word_num(self):
#         return self._word_num
    
#     @word_num.setter
#     def word_num(self, value):
#         logger = logging.getLogger(__name__ + ".Top2VecTopicPart[" + str(self.key) + "].word_num")
#         logger.info("Starting")
#         if len(self.word_list) < value:
#             self.word_list.clear()
#             try:
#                 # Retrieve the Top2Vec model from the parent object
#                 top2vec_model = self.parent.model # Assuming this is how you access the Top2Vec model
                
#                 # Get topics and keywords from the Top2Vec model
#                 topics, probs, _ = top2vec_model.get_topics(num_topics=value)
                
#                 # Iterate over topics and add top words to word_list
#                 for topic_idx, topic in enumerate(topics):
#                     word_prob_list = list(zip(topic, probs[topic_idx]))
#                     self.word_list.extend(word_prob_list)
#             except Exception as e:
#                 print("Error:", e)
#                 logger.error("Error occurred: %s", e)
#         self._word_num = value
#         logger.info("Finished")
            
# class Top2VecSample(TopicSample):
#     def __init__(self, name, dataset_key, model_parameters):
#         TopicSample.__init__(self, name, dataset_key, "Top2Vec", model_parameters)

#         #fixed properties that may be externally accessed but do not change after being initialized

#         #these need to be removed before pickling during saving due to threading and use of multiple processes
#         #see __getstate__ for removal and Load and Reload for readdition
#         self.training_thread = None
#         self.vectorizer = None
#         self.transformed_texts = None
#         self.model = None

#     def __getstate__(self):
#         state = dict(self.__dict__)
#         state['training_thread'] = None
#         state['vectorizer'] = None
#         state['transformed_texts'] = None
#         state['model'] = None
#         return state

#     def __repr__(self):
#         return 'Top2VecSample[%s][%s]' % (self.name, self.key,)
    
#     def GenerateStart(self, notify_window, current_workspace_path, start_dt):
#         logger = logging.getLogger(__name__+"."+repr(self)+".GenerateStart")
#         logger.info("Starting")
#         self.start_dt = start_dt
#         self.training_thread = SamplesThreads.Top2VecTrainingThread(notify_window,
#                                                                    current_workspace_path,
#                                                                    self.key,
#                                                                    self.tokensets,
#                                                                    self.num_topics)
#         logger.info("Finished")
    
#     def GenerateFinish(self, result, dataset, current_workspace):
#         logger = logging.getLogger(__name__+"."+repr(self)+".GenerateFinish")
#         logger.info("Starting")
#         self.generated_flag = True
#         self.training_thread.join()
#         self.training_thread = None
#         self._tokensets = list(self.tokensets.keys())
#         with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
#            self.vectorizer = pickle.load(infile)
#         with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
#             self.transformed_texts = pickle.load(infile)
#         with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/top2vec_model.pk', 'rb') as infile:
#             self.model = pickle.load(infile)

#         self.document_topic_prob = result['document_topic_prob']

#         for i in range(self.num_topics):
#             topic_num = i+1
#             self.parts_dict[topic_num] = Top2VecTopicPart(self, topic_num, dataset)
#         self.parts_dict['unknown'] = TopicUnknownPart(self, 'unknown', [], dataset)

#         self.word_num = 10
#         self.ApplyDocumentCutoff()
        
#         self.end_dt = datetime.now()
#         logger.info("Finished")

#     def OldLoad(self, workspace_path):
#         logger = logging.getLogger(__name__+"."+repr(self)+".Load")
#         logger.info("Starting")
#         self._workspace_path = workspace_path
#         if self.generated_flag:
#             with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf_vectorizer.pk', 'rb') as infile:
#                 self.vectorizer = pickle.load(infile)
#             with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf.pk', 'rb') as infile:
#                 self.transformed_texts = pickle.load(infile)
#             with bz2.BZ2File(self._workspace_path+self.filedir+'/top2vec_model.pk', 'rb') as infile:
#                 self._model = pickle.load(infile)
#         logger.info("Finished")

#     def Load(self, current_workspace):
#         logger = logging.getLogger(__name__+"."+repr(self)+".Load")
#         logger.info("Starting")
#         if self.generated_flag:
#             with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
#                 self.vectorizer = pickle.load(infile)
#             with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
#                 self.transformed_texts = pickle.load(infile)
#             with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/top2vec_model.pk', 'rb') as infile:
#                 self._model = pickle.load(infile)
#         logger.info("Finished")

#     def Save(self, current_workspace):
#         logger = logging.getLogger(__name__+"."+repr(self)+".Save")
#         logger.info("Starting")
#         if self.vectorizer is not None:
#             with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
#                 pickle.dump(self.vectorizer, outfile)
#         if self.transformed_texts is not None:
#             with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
#                 pickle.dump(self.transformed_texts, outfile)
#         if self.model is not None:
#             with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/top2vec_model.pk', 'wb') as outfile:
#                 pickle.dump(self.model, outfile)
#         logger.info("Finished")

# #New changes here 
# class Top2VecModelCreateDialog(wx.Dialog):
#     def __init__(self, parent):
#         logger = logging.getLogger(__name__+".Top2VecModelCreateDialog.__init__")
#         logger.info("Starting")
#         wx.Dialog.__init__(self, parent, title=GUIText.CREATE_TOP2VEC)

#         self.model_parameters = {}

#         sizer = wx.BoxSizer(wx.VERTICAL)

#         #need to only show tokensets that have fields containing data
#         self.usable_datasets = []
#         main_frame = wx.GetApp().GetTopWindow()
#         for dataset in main_frame.datasets.values():
#             if len(dataset.computational_fields) > 0:
#                 self.usable_datasets.append(dataset.key)
#         if len(self.usable_datasets) > 1: 
#             dataset_label = wx.StaticText(self, label=GUIText.DATASET+":")
#             usable_datasets_strings = [str(dataset_key) for dataset_key in self.usable_datasets]
#             self.dataset_ctrl = wx.Choice(self, choices=usable_datasets_strings)
#             dataset_sizer = wx.BoxSizer(wx.HORIZONTAL)
#             dataset_sizer.Add(dataset_label, 0, wx.ALL|wx.ALIGN_CENTRE_VERTICAL, 5)
#             dataset_sizer.Add(self.dataset_ctrl, 0, wx.ALL, 5)
#             sizer.Add(dataset_sizer)

#         num_topics_label = wx.StaticText(self, label=GUIText.NUMBER_OF_TOPICS_CHOICE)
#         self.num_topics_ctrl = wx.SpinCtrl(self, min=1, max=10000, initial=10)
#         self.num_topics_ctrl.SetToolTip(GUIText.NUMBER_OF_TOPICS_TOOLTIP)
#         num_topics_sizer = wx.BoxSizer(wx.HORIZONTAL)
#         num_topics_sizer.Add(num_topics_label, 0, wx.ALL|wx.ALIGN_CENTRE_VERTICAL, 5)
#         num_topics_sizer.Add(self.num_topics_ctrl, 0, wx.ALL, 5)
#         sizer.Add(num_topics_sizer)

#         num_passes_label = wx.StaticText(self, label=GUIText.NUMBER_OF_PASSES_CHOICE)
#         # Number of passes is min 100, max 1000, that's why app gets crashed if pass given <100 ! 
#         self.num_passes_ctrl = wx.SpinCtrl(self, min=1, max=1000, initial=100)
#         self.num_passes_ctrl.SetToolTip(GUIText.NUMBER_OF_PASSES_TOOLTIP)
#         num_passes_sizer = wx.BoxSizer(wx.HORIZONTAL)
#         num_passes_sizer.Add(num_passes_label, 0, wx.ALL|wx.ALIGN_CENTRE_VERTICAL, 5)
#         num_passes_sizer.Add(self.num_passes_ctrl, 0, wx.ALL, 5)
#         sizer.Add(num_passes_sizer)

#         controls_sizer = self.CreateButtonSizer(wx.OK|wx.CANCEL)
#         ok_button = wx.FindWindowById(wx.ID_OK, self)
#         ok_button.SetLabel(GUIText.CREATE_TOP2VEC)
#         ok_button.Bind(wx.EVT_BUTTON, self.OnOK, id=wx.ID_OK)
#         sizer.Add(controls_sizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

#         self.SetSizer(sizer)
        
#         self.Layout()
#         self.Fit()
#         logger.info("Finished")

#     def OnOK(self, event):
#         logger = logging.getLogger(__name__+".Top2VecModelCreateDialog.OnOK")
#         logger.info("Starting")
#         main_frame = wx.GetApp().GetTopWindow()
#         #check that name exists and is unique
#         status_flag = True

#         main_frame.model_iter += 1
#         model_name = "Model_"+str(main_frame.model_iter)

#         if len(self.usable_datasets) > 1:
#             dataset_id = self.dataset_ctrl.GetSelection()
#             if dataset_id is wx.NOT_FOUND:
#                 wx.MessageBox(GUIText.DATASET_MISSING_ERROR,
#                             GUIText.ERROR, wx.OK | wx.ICON_ERROR)
#                 logger.warning('dataset was not chosen')
#                 status_flag = False
#         elif len(self.usable_datasets) == 1:
#             dataset_id = 0
#         else:
#             wx.MessageBox(GUIText.DATASET_NOTAVAILABLE_ERROR,
#                           GUIText.ERROR, wx.OK | wx.ICON_ERROR)
#             logger.warning('no dataset available')
#             status_flag = False

#         if status_flag:
#             self.model_parameters['name'] = model_name
#             self.model_parameters['dataset_key'] = self.usable_datasets[dataset_id]
#             self.model_parameters['num_topics'] = self.num_topics_ctrl.GetValue()
#             self.model_parameters['num_passes'] = self.num_passes_ctrl.GetValue()
#             self.model_parameters['alpha'] = None
#             self.model_parameters['eta'] = None
#         logger.info("Finished")
#         if status_flag:
#             self.EndModal(wx.ID_OK)

 # #New changes here 
        # elif model_type == 'Top2Vec':
        #     main_frame.PulseProgressDialog(GUIText.GENERATING_TOP2VEC_MSG2)
        #     new_sample = Samples.Top2VecSample(name, dataset_key, model_parameters)
        #     new_sample.fields_list = fields_list
        #     new_sample.applied_filter_rules = copy.deepcopy(dataset.filter_rules)
        #     new_sample.tokenization_choice = dataset.tokenization_choice
        #     new_sample.tokenization_package_versions = dataset.tokenization_package_versions
        #     new_sample_panel = TopicSamplePanel(parent_notebook, new_sample, dataset, self.GetParent().GetSize())  
        #     main_frame.samples[new_sample.key] = new_sample
        #     new_sample.GenerateStart(new_sample_panel, main_frame.current_workspace.name, self.start_dt)
        #     main_frame.StepProgressDialog(GUIText.GENERATING_TOP2VEC_MSG3)
        #     parent_notebook.InsertPage(len(parent_notebook.sample_panels), new_sample_panel, new_sample.name, select=True)
        #     parent_notebook.sample_panels[new_sample.key] = new_sample_panel
        #     main_frame.CloseProgressDialog(message=GUIText.GENERATED_TOP2VEC_COMPLETED_PART1, thaw=False)

                # elif sample.sample_type == "Top2Vec":
                #     new_sample_panel = SamplesGUIs.TopicSamplePanel(self, sample, main_frame.datasets[sample.dataset_key], size=self.GetSize())
                #     new_sample_panel.Load({})

 # #New changes here            
        # elif model_type == 'Top2Vec':
        #     with Top2VecModelCreateDialog(self) as create_dialog:
        #         if create_dialog.ShowModal() == wx.ID_OK:
        #             self.Freeze()
        #             main_frame.CreateProgressDialog(GUIText.GENERATING_DEFAULT_LABEL,
        #                                              warning=GUIText.GENERATE_WARNING+"\n"+GUIText.SIZE_WARNING_MSG,
        #                                              freeze=False)
        #             main_frame.StepProgressDialog(GUIText.GENERATING_DEFAULT_MSG)
        #             self.start_dt = datetime.now()
        #             model_parameters = create_dialog.model_parameters
        #             name = model_parameters['name']
        #             main_frame.PulseProgressDialog(GUIText.GENERATING_TOP2VEC_SUBLABEL+str(name))
        #             self.capture_thread = SamplesThreads.CaptureThread(self,
        #                                                                main_frame,
        #                                                                model_parameters,
        #                                                                model_type)

 # #New changes here 
        # create_top2vec_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # topicmodel_sizer.Add(create_top2vec_sizer)
        # create_top2vec_button = wx.Button(self, label=GUIText.TOP2VEC_LABEL)
        # create_top2vec_button.SetToolTip(GUIText.CREATE_TOP2VEC_TOOLTIP)
        # self.Bind(wx.EVT_BUTTON, lambda event: self.OnStartCreateSample(event, 'Top2Vec'), create_top2vec_button)
        # create_top2vec_sizer.Add(create_top2vec_button, 0, wx.ALL, 5)
        # create_top2vec_description = wx.StaticText(self, label=GUIText.TOP2VEC_DESC)
        # create_top2vec_sizer.Add(create_top2vec_description, 0, wx.ALL|wx.ALIGN_CENTRE_VERTICAL, 5)
        # create_top2vec_link = wx.adv.HyperlinkCtrl(self, label="5", url=GUIText.TOP2VEC_URL) 
        # create_top2vec_sizer.Add(create_top2vec_link, 0, wx.ALL|wx.ALIGN_CENTRE_VERTICAL, 5)


    
