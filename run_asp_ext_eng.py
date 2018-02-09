

# this is the code needed for running the sentiment analysis engine
import pandas as pd
import re
from langdetect import detect
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
#from textblob import TextBlob,
from textblob import Word, Blobber
import textblob_fr
import textblob_de
import os
import shutil # from removing files from one folder to another
os.environ["NLS_LANG"] = ".AL32UTF8" 
import sys
reload(sys)
sys.setdefaultencoding('utf8')

############
## the directories shown below should be changed when the new folder for the aspect extraction is given
############

import os
input_directory='/home/ayaqubov/engines/Windows-share/Input/'
files_in_folder=os.listdir(input_directory)
output_directory='/home/ayaqubov/engines/Windows-share/Output/'
processed_directory='/home/ayaqubov/engines/Windows-share/Processed/'


# check while the input is empty do the 

#while(os.listdir('/home/ayaqubov/engines/Windows-share/Input/')!=""):
# get the data and process the data

#functions to process the data:
def count_words(sentence):
    words=word_tokenize(sentence)
    l=len(words)
    return l

import sys
sys.path.insert(0, '/home/ayaqubov/sentvec_new/italian/')
from fasttext import FastVector
en_model=FastVector(vector_file='/home/ayaqubov/sentvec_new/fasttext_embeddings/eng/wiki.en.vec')
import numpy as np
def find_min_distance(word,list_of_categories):
    l=len(list_of_categories)
    distances=[]
    for i in range(0,l):
        a=en_model[word]
        b=en_model[list_of_categories[i]]
        d=np.linalg.norm(a-b)
        distances.append(d)
    return distances.index(min(distances))

def get_sentiment_en(sentence):
    from textblob import TextBlob
    # this function is the simplest function
    blob=TextBlob(sentence)
    sentiment_pol=blob.sentiment.polarity
    #sentiment_sub=TextBlob(sentence).sentiment.subjectivity
    #sentiment_=sentiment_pol
    return sentiment_pol
from textblob_fr import PatternTagger, PatternAnalyzer
from textblob import Blobber
def get_sentiment_fr(sentence):
    mysentence=str(sentence)
    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    sentim = tb(mysentence).sentiment
    sentiment_pol=sentim[0]
    sentiment_sub=sentim[1]
    return sentiment_pol

from textblob_de import TextBlobDE as TextBlob
from textblob_de import PatternTagger
def get_sentiment_de(sentence):
    blob = TextBlob(sentence)
    sentim=blob.sentiment
    sentiment_pol=sentim[0]
    sentiment_sub=sentim[1]
    return sentiment_pol
import langid
def identify_lang(sentence):
    cl=langid.classify(sentence)
    lan=cl[0]
    return lan
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
def stem_word(in_word):
    stemmed_word=ps.stem(in_word)
    return stemmed_word

import langid
def identify_lang(sentence):
    cl=langid.classify(sentence)
    lan=cl[0]
    return lan
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
def stem_word(in_word):
    stemmed_word=ps.stem(in_word)
    return stemmed_word



### aspect funcs.
### all these functions below make up the aspect extraction from the sentence

## This file contains functions needed to extract the aspects
def word_tokens(mysentence):
    # tokenization
    import nltk
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    #mysent='This camera is sleek, heavy and very affordable.' #I liked the book. Not to mention the price of the phone. '
    #mysent='Love the sleekness of the player.'
    #mysent='Very big to hold'
    #mysent='Not to mention the price of the phone.'
    #mysent='We ordered chicken casserole, but what we got were a few small pieces of chicken, all dark meat and on the bone.'
    #mysentence='Logitech mouse was nice.'
    mywords=tokenizer.tokenize(mysentence)
    tags=nltk.pos_tag(mywords)
    return tags

def do_stemming(word):
    from nltk.stem.lancaster import LancasterStemmer
    st = LancasterStemmer()
    stemmed=st.stem(word)
    return stemmed

def dependency_parse_sentence(mysentence):
    from nltk.parse.stanford import StanfordDependencyParser
    path_to_jar='/home/ayaqubov/stanford_nlp/stanford_parser/stanford-parser.jar'
    path_to_models_jar='/home/ayaqubov/stanford_nlp/stanford_parser/stanford-parser-3.5.2-models.jar'
    dependency_parser=StanfordDependencyParser(path_to_jar=path_to_jar,path_to_models_jar=path_to_models_jar)
    result = dependency_parser.raw_parse(mysentence)
    dep = result.next()
    dependency_results=list(dep.triples())
    #print "Dependency results:::"
    #print dependency_results
    return dependency_results


def extract_explicit(mysentence):
    ### finding explicit aspects based on the rules
    dependency_results=dependency_parse_sentence(mysentence)
    aspects_explicit=[]
    ld=len(dependency_results)
    for i in range(0,ld):
        #rule 1
        if(dependency_results[i][1] in ['amod','nmod','nummod','appos','det','nsubjpass']):# among nominal modifiers of the noun
            if(dependency_results[i][2][1] in ['NN','NNP']):
                asp=dependency_results[i][2][0]
                aspects_explicit.append(asp)

        # rule 1
        if(dependency_results[i][1] in ['nsubj','dobj','iobj']): #adverbial or adjective modifier ---clausal argument relations
            if(dependency_results[i][2][1] in ['NN','NNP'] and dependency_results[i][0][1] in ['JJ','JJR','JJS','RB','RBR','RBS']):
                asp=dependency_results[i][2][0]
                aspects_explicit.append(asp)

        # rule 2.
        if(dependency_results[i][1] in ['nsubj','dobj','iobj']):
            if(dependency_results[i][2][1] in ['NN','NNP'] and dependency_results[i][0][1] in ['VB','VBD','VBG','VBP','VBZ']):
                asp=dependency_results[i][2][0]
                aspects_explicit.append(asp)


       

    
        if(dependency_results[i][1]=='cop'):
            if(dependency_results[i][0][1] in ['NNP','NNPS']):
                aspects_explicit.append(dependency_results[i][0][1])
    
    
        # rule 3

    ##### sentences which do not have subject noun relation in their parse tree
    #rule 3.3.4 .2
    for ii in range(0,ld):
        if(dependency_results[ii][1]=='case'):
            mynoun=dependency_results[ii][0][0]
            for kk in range(0,ld):
                if(dependency_results[kk][2][0]==mynoun and dependency_results[kk][0][1] in ['NN','NNS','NNP','NNPS']):
                    aspects_explicit.append(dependency_results[kk][0][0])
                    if(dependency_results[ii][0][1] !='PRP'):
                        aspects_explicit.append(mynoun)

    ### among additional rules:
    # rule 3.3.5. 2
    for iii in range(0,ld):
        if(dependency_results[iii][1]=='compound'):
            if(dependency_results[iii][0][1] in ['NN','NNS','NNP','NNPS'] and dependency_results[iii][2][1] in ['NN','NNS','NNP','NNPS']):
                new_aspect=dependency_results[iii][2][0]+' '+dependency_results[iii][0][0]
                aspects_explicit.append(new_aspect)
    unique_explicit_aspects=list(set(aspects_explicit))     
    return unique_explicit_aspects

def extract_implicit(mysentence):
    dependency_results=dependency_parse_sentence(mysentence)
    print dependency_results
    ## getting the implicit aspects

    implicit_aspect_categories=['functionality','weight','price','appearance','behaviour','performance','qualtiy','service','size']

    implicit_aspects=[]
    dl=len(dependency_results)
    for i in range(0,dl):
        if(dependency_results[i][1]=='cop'):## if the token is in copular relation with the copular verb and copular verb exists in implicit aspect lexicon
            if(dependency_results[i][0][1] not in ['NNP','NNPS']):
                implicit_indicator=dependency_results[i][0][0]
                implicit_aspects.append(implicit_indicator)

                for kk in range(0,len(dependency_results)):
                    if(dependency_results[kk][0][0]==implicit_indicator and dependency_results[kk][1]=='conj'):
                        #implicit_indicator=dependency_results[kk][2][0]
                        implicit_aspects.append(dependency_results[kk][2][0])

                    if(dependency_results[kk][0][0]==implicit_indicator and dependency_results[kk][2][1]=='VB'):
                        implicit_aspects.append(dependency_results[kk][2][0])
                        
                        
         ## check for copular verbs
        copular_verbs=['is', 'am', 'are', 'was', 'were', 'appear', 'seem', 'look', 'sound', 'smell', 'taste', 'feel', 'become', 'get']
        ####Use copular verbs
        #if the verb is modified by adjective and adverb:
        if(dependency_results[i][1] in ['nsubj','dobj','iobj','ccomp','xcomp'] ):
            if(dependency_results[i][0][1] in ['VB','VBD','VBG','VBP','VBZ'] and dependency_results[i][2][1] in ['JJ','JJR','JJS','RB','RBR','RBS']):
                #if the noun is in subject relation verb
                #take that verb:
                myverb=dependency_results[i][0][0]
                # search for the noun in that sentence
                
                stem_verb=do_stemming(myverb)
                implicit_aspects.append(stem_verb)

        if(dependency_results[i][1]=='amod'):
            if(dependency_results[i][2][1]=='JJ' and dependency_results[i][0][1] in ['NN','NNS']):
                implicit_aspects.append(dependency_results[i][2][0])
                
        if(dependency_results[i][1]=='nsubjpass'):
            if(dependency_results[i][0][1] in ['JJ','JJR','JJS'] and dependency_results[i][2][1] in ['NN','NNS']):
                implicit_aspects.append(dependency_results[i][0][0])

    import gensim
    from gensim.models import word2vec
    
    # get the pretrained word2vec model 
    #fname="/home/ayaqubov/AspectExtractor/myword2vec/word2vecmodel_electronics"
    #mymodel.save(fname)
    # To get back the model
    #mymodel = gensim.models.Word2Vec.load(fname)
    
    #en_model=FastVector(vector_file='/home/ayaqubov/sentvec_new/fasttext_embeddings/eng/wiki.en.vec')
    #mymodel
    #en_model=FastVector(vector_file='/home/ayaqubov/sentvec_new/fasttext_embeddings/eng/wiki.en.vec')
    

    # remove non-frequent implicit aspects from the sentences --->>> this is needed because we want to compare these to the categories
    unique_implicit_aspects=list(set(implicit_aspects))
    print 'Implicit aspects here:::'
    print unique_implicit_aspects
    
    ### mapping of implicit aspect categories
    implicit_aspect_categories=['functionality','weight','price','appearance','behaviour','performance','qualtiy','service','size']
    list2=[]
    
    
    output_categories=[]
    for k in range(0,len(unique_implicit_aspects)):
        asp_=unique_implicit_aspects[k]
        
        try:
            a=find_min_distance(asp_,implicit_aspect_categories)
            output_categories.append(implicit_aspect_categories[a])
        except:
            print "Implicit aspect directly goes to output ... Model does not contain the word."
            output_categories.append(asp_)
            
        
    
    return output_categories




def importance_of_aspects(aspects_explicit):
    #printing the importance of the explicit aspects
    from collections import Counter
    counter_aspects=Counter(aspects_explicit)
    print(counter_aspects)

def find_max_index(mylist):
        import numpy as np
        ind = np.argmax(mylist)
        return ind

## this is with different kind of model
# def extract_implicit(mysentence):
#     dependency_results=dependency_parse_sentence(mysentence)
#     print dependency_results
#     ## getting the implicit aspects

#     implicit_aspect_categories=['functionality','weight','price','appearance','behaviour','performance','qualtiy','service','size']

#     implicit_aspects=[]
#     dl=len(dependency_results)
#     for i in range(0,dl):
#         if(dependency_results[i][1]=='cop'):## if the token is in copular relation with the copular verb and copular verb exists in implicit aspect lexicon
#             if(dependency_results[i][0][1] not in ['NNP','NNPS']):
#                 implicit_indicator=dependency_results[i][0][0]
#                 implicit_aspects.append(implicit_indicator)

#                 for kk in range(0,len(dependency_results)):
#                     if(dependency_results[kk][0][0]==implicit_indicator and dependency_results[kk][1]=='conj'):
#                         #implicit_indicator=dependency_results[kk][2][0]
#                         implicit_aspects.append(dependency_results[kk][2][0])

#                     if(dependency_results[kk][0][0]==implicit_indicator and dependency_results[kk][2][1]=='VB'):
#                         implicit_aspects.append(dependency_results[kk][2][0])
                        
                        
#          ## check for copular verbs
#         copular_verbs=['is', 'am', 'are', 'was', 'were', 'appear', 'seem', 'look', 'sound', 'smell', 'taste', 'feel', 'become', 'get']
#         ####Use copular verbs
#         #if the verb is modified by adjective and adverb:
#         if(dependency_results[i][1] in ['nsubj','dobj','iobj','ccomp','xcomp'] ):
#             if(dependency_results[i][0][1] in ['VB','VBD','VBG','VBP','VBZ'] and dependency_results[i][2][1] in ['JJ','JJR','JJS','RB','RBR','RBS']):
#                 #if the noun is in subject relation verb
#                 #take that verb:
#                 myverb=dependency_results[i][0][0]
#                 # search for the noun in that sentence
                
#                 stem_verb=do_stemming(myverb)
#                 implicit_aspects.append(stem_verb)

#         if(dependency_results[i][1]=='amod'):
#             if(dependency_results[i][2][1]=='JJ' and dependency_results[i][0][1] in ['NN','NNS']):
#                 implicit_aspects.append(dependency_results[i][2][0])
                
#         if(dependency_results[i][1]=='nsubjpass'):
#             if(dependency_results[i][0][1] in ['JJ','JJR','JJS'] and dependency_results[i][2][1] in ['NN','NNS']):
#                 implicit_aspects.append(dependency_results[i][0][0])

#     import gensim
#     from gensim.models import word2vec

#     # get the pretrained word2vec model 
#     fname="/home/ayaqubov/AspectExtractor/myword2vec/word2vecmodel_electronics"
#     #mymodel.save(fname)
#     # To get back the model
#     mymodel = gensim.models.Word2Vec.load(fname)
    
#     # remove non-frequent implicit aspects from the sentences --->>> this is needed because we want to compare these to the categories
#     unique_implicit_aspects=list(set(implicit_aspects))
#     print 'Implicit aspects here:::'
#     print unique_implicit_aspects
    
#     ### mapping of implicit aspect categories
#     implicit_aspect_categories=['functionality','weight','price','appearance','behaviour','performance','qualtiy','service','size']
#     list2=[]

#     for i in range(0,len(unique_implicit_aspects)):
#         list2.append(unique_implicit_aspects[i])

#     similarities=[]
#     for word2 in list2:
#         s=[]
#         for asp_cat in implicit_aspect_categories:
#             try:
#                 s.append(mymodel.similarity(word2,asp_cat))
#             except:
#                 print("This word was not found -- improvement with fasttext")
#                 s.append(word2)
#         similarities.append(s)

#     max_indices=[]
#     for i in range(0,len(similarities)):
#         m=find_max_index(similarities[i])
#         max_indices.append(m)

#     # this part does aspect category paring using the results from the WORD2VEC
#     aspect_category_pair = dict()
#     for i in range(0,len(unique_implicit_aspects)):
#         max_l=max_indices[i]
#         aspect_category_pair[unique_implicit_aspects[i]]=implicit_aspect_categories[max_l]

#     #combining implicit and explicit aspects
#     #total_aspects=aspects_explicit
#     implicit_categories=aspect_category_pair.values()
#     return implicit_categories


def combine_aspects(aspects_explicit,implicit_categories):
    total_aspects=aspects_explicit
    
    for ij in range(0,len(implicit_categories)):
        total_aspects.append(implicit_categories[ij])
    #here are the total aspects
    # these aspects represnet the sentence
    #print(total_aspects)
    #now issue is to find the overall 
    #return total_aspects
    #total_aspects=aspects_explicit
    stotal_aspects=list(set(total_aspects))
    return stotal_aspects



#####################
#2 ways to run : either run the main function or run the get_aspects function

########################

def get_aspects(mysentence):
    import os
    from nltk.parse import stanford
    import os
    java_path = "/usr/lib/jvm/jre-1.8.0"
    os.environ['JAVAHOME'] = java_path
    myimplicit=extract_implicit(mysentence)
    myexplicit=extract_explicit(mysentence)
    total_aspects=combine_aspects(myexplicit,myimplicit)
    if(total_aspects==[]): # when we dont have specific aspect in the sentence, we add 'GENERAL'
        total_aspects.append('GENERAL')
    return total_aspects



for ifiles in range(0,len(files_in_folder)):
    file_directory=input_directory + files_in_folder[ifiles]
    print(file_directory)
    #file_directory
    mycolumnsdf=['REVIEW_WID','REVIEW_FULL_TEXT']
    df_aws_reviews=pd.DataFrame(columns=mycolumnsdf)
    df_aws_reviews=pd.read_csv(file_directory)
    df_aws_reviews['LANGUAGE']=""
    num_unique_reviews=df_aws_reviews.shape[0]

    ###########################################################
    # here comes the check the language part and write correspondingly to the df_aws_reviews dataframe

    from nltk.tokenize import sent_tokenize, word_tokenize
    ## data processing
    print('Processing...')
    index=0
    
    for u in range(0,num_unique_reviews):
        review_id=df_aws_reviews.iloc[u,0]
        print(review_id)
        review_text=df_aws_reviews.iloc[u,1]
        review_str=str(review_text)
        #review_strw=review_str.encode('cp1252')## windows encoded
        #ureview_str=review_strw.decode('utf-8')
        #ureview_str=unicode(review_str,"utf-8")
        ureview_str=review_str.decode('cp1252').encode('utf-8',errors='ignore')
        #print(review_str)
        #print(len(review_str))
        contain_l=re.search('[a-zA-Z]', review_str)
        if(contain_l!='None'):
            # handle japanese,arabic,chinese cases because they appear as the ?? marks in the results
            try:
                text_lang=detect(ureview_str)
                #text_lang2=identify_lang(review_str)
            except:
                print 'Error in reading, keep reading'
                #i_debug+=1
                continue
            #check_words=word_tokenize(review_str)
            #for iiii in range(0,len(check_the_words)):
            #    if (check_the_words[iiii]=='product' or check_the_words[iiii]=='excellent'  or check_the_words[iiii]=='mouse'):
            #        break
            #            #continue

            if(text_lang=='en'):
                df_aws_reviews.iloc[u,2]='English'
                index+=1

            if(text_lang=='de'):
                df_aws_reviews.iloc[u,2]='German'
                index+=1

            if(text_lang=='fr'):
                if('product' in ureview_str or 'excellent' in ureview_str or 'mouse' in ureview_str):
                    df_aws_reviews.iloc[u,2]='English'
                else:
                    df_aws_reviews.iloc[u,2]='French'
                index+=1




    # delete reviews that does not have language information
    # can also use the drop function from the pandas library
    df_aws_reviews=df_aws_reviews[df_aws_reviews['LANGUAGE'] != ""]
    df_aws_reviews.shape

    


    #########################################################
    # division into the sentences
    df_num_of_rows=df_aws_reviews.shape[0]
    #mycolumns=['REVIEWSENTENCE_WID','REVIEW_WID','SENTENCE_ID','SKU','COUNTRY','SITE_URL','REVIEW_POSTED_DATE','WORD_COUNT',
    #                    'SENTIMENT','STAR_RATING','SENTENCE','PRODUCT_TYPE','PRODUCT_GROUP','PRODUCT_LINE_NAME']
    mycolumns_sentence=['textsentence_id','text_id','sentence_id','word_count',
                        'Aspects','sentence','language']
    df_aws_sentences=pd.DataFrame(columns=mycolumns_sentence)

    # adding sentences
    index_=0
    for i in range(0,df_num_of_rows):
        print("i is ",i)
        this_review=df_aws_reviews.iloc[i,1]
            # use try except because errpr was occuring in some cases
        try:
            sentences_this_review=sent_tokenize(this_review)
        except:
            print("Error in sentence tokenizing")
            continue
        num_of_sents=len(sentences_this_review)
        current_review_id=str(df_aws_reviews.iloc[i,0])
        #print(current_review_id)
        if(num_of_sents!=0):
            sent_id=0
            for j in range(0,num_of_sents):
                current_sentence=sentences_this_review[j]
                if(current_sentence in ["!","?","."]):
                    continue
                word_count=count_words(current_sentence)
                reviewsentence_id=current_review_id+'_'+str(sent_id)#int(current_review_id+'_'+str(sent_id))
                # Now calculate the polarity of sentece:
                #sentiment_=get_sentiment(current_sentence)
                if(df_aws_reviews.iloc[i,2]=='English'):
                    aspects_extracted=get_aspects(current_sentence)
                else:
                    print('Does not support other languages than English...')
                ##### since some aspects may be repeated, we do the following:
                ### we make the dictionary out of aspects and put that in the cell corresponding to the
                ### or we just get the set values:
                print(aspects_extracted)
                saspects_extracted=list(set(aspects_extracted)) ## this will contain aspects only once.
                
                one_row=[reviewsentence_id,current_review_id,sent_id,word_count,saspects_extracted,current_sentence,df_aws_reviews.iloc[i,2]]
                df_aws_sentences.loc[index_]=one_row
                sent_id+=1
                index_+=1

                
 
    df_aws_sentences[['sentence_id','word_count']]=df_aws_sentences[['sentence_id','word_count']].astype(int)
    ########################################################
    ## now word frequency table

    cols_word_freq= ['reviewsentence_wid','review_wid','sentence_id','word','translated_word','freq']
    df_sents_num_of_rows=df_aws_sentences.shape[0]


    df_word_freq=pd.DataFrame(columns=cols_word_freq)
    # get rid of commas etc
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    import re
    def check_num(input_s): 
        num_format = re.compile("^[\-]?[1-9][0-9]*\.?[0-9]+$")
        isnumber = re.match(num_format,input_s)
        #isnumber=~
        if isnumber:
            return True
        else:
            return False
    def check_letter(input_s):
        #remove len 1 and 2s(come back for len 2 later)
        l=len(input_s)
        if(l==1 or l==2):
            return True
        return False

    from nltk.corpus import stopwords
    #stopwords_ = set(stopwords.words('english'))
    stopwords_fr=set(stopwords.words('french'))
    stopwords_en=set(stopwords.words('english'))
    stopwords_ge=set(stopwords.words('german'))


    windex_=0

    for i in range(0,df_sents_num_of_rows):
        #print(i)
        sentence=df_aws_sentences.iloc[i,5]
        #words=word_tokenize(sentence)
        # maybe use try-except block as follows:
        #try:
        #words=tokenizer.tokenize(sentence)
        #except:
        #print("error in word tokenizing")
        words=tokenizer.tokenize(sentence)
        tags_=nltk.pos_tag(words)
        num_words=len(words)
        for j in range(0,num_words):
            word=words[j]
            wordlow=word.lower()
            # check if it is noun here
            translated=wordlow  ## for another language we need translation
            freq=1
            w_isnum=check_num(wordlow)
            one_two_let=check_letter(wordlow)
            if(df_aws_sentences.iloc[i,6]=='English'):
                stopwords_=stopwords_en
            if(df_aws_sentences.iloc[i,6]=='French'):
                stopwords_=stopwords_fr
            if(df_aws_sentences.iloc[i,6]=='German'):
                stopwords_=stopwords_ge
            if(wordlow in stopwords_ or w_isnum or one_two_let):
                continue


            if(tags_[j][1]=='NN' or tags_[j][1]=='NNS' or tags_[j][1]=='NNP'):
                # since we expect aspects are more likely to be among the nouns

                one_row=[df_aws_sentences.iloc[i,0],df_aws_sentences.iloc[i,1],df_aws_sentences.iloc[i,2],wordlow,translated,freq]
                df_word_freq.loc[windex_]=one_row
                windex_+=1
            #print(windex_)

    df_word_freq[['sentence_id','freq']]=df_word_freq[['sentence_id','freq']].astype(int)
    df_word_freq.rename(columns={'reviewsentence_wid':'textsentence_id','review_wid':'text_id'},inplace=True)
    #######################################################
    # give indices a name
    #df.index.rename('Index')
    
    ############################################################################
    ## output file generation and moving the Input file from Processed part
    ## this bassically will be what the users need
    sentences_name_to_save=files_in_folder[ifiles][:-4]+'_ouput.csv'
    output_sentence_directory=output_directory+sentences_name_to_save
    df_aws_sentences.to_csv(output_sentence_directory,index=False)


    ###################################
    # merge 2 tables  -- in this part
    df_merged_output=df_aws_sentences.merge(df_word_freq,how='left',on=['textsentence_id','text_id','sentence_id'])
    merged_name_to_save=files_in_folder[ifiles][:-4]+'_output_aspect.csv'
    merged_output_directory=output_directory+merged_name_to_save
    df_merged_output.to_csv(merged_output_directory,index=False)

    # move processed file into folder named 
    shutil.move(file_directory,processed_directory)


    ## 2 output files are generated -- 
    # 1. sentences with sentiments
    # 2. merged table which contains word frequency-- this maybe used for visualisation in Tableau