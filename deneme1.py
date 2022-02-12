
import String_Similarity


dosya = open('sorular.txt')
document=dosya.readlines()
dosya.close()

query=input("Sorunuzu giriniz: ")
document.append(query)
vector_of_doc=String_Similarity.vector_of_document(document)

tf_vectors=([])
tf_idf_vectors=([])

i=0
for i in range(0,len(document)):
    tf_vectors.append(String_Similarity.tf_vector(vector_of_doc, document[i]))

idf_vector_of_doc=String_Similarity.idf_vector_of_document(vector_of_doc,tf_vectors)


max_tf=max(max(tf_vectors))
tf_vectors=String_Similarity.normalize_tf_vectors(tf_vectors,max_tf)

i=0
for i in range(0,len(tf_vectors)):
    tf_idf_vectors.append(String_Similarity.tf_idf_vector(tf_vectors[i], idf_vector_of_doc))


query_index = len(document) - 1

min=2
i=0
for i in range(0,len(tf_vectors)-2):
    tmp_js=String_Similarity.jensen_shanon_rate(tf_vectors[i],tf_vectors[query_index])
    if(min>tmp_js):
        min=tmp_js
        tmp_i=i

print(query, " sorusuna en benzer soru jensen shanon metoduna göre" , tmp_js , " oranı ile " , tmp_i ,". soru:" , document[tmp_i])

max=0
i=0
for i in range(0,len(document)-2):
    tmp_jk=String_Similarity.jakard_similarity_rate(document[i],document[query_index])
    if(max<tmp_jk):
        max=tmp_jk
        tmp_i=i

print(query, " sorusuna en benzer soru jakard metoduna göre" , tmp_jk , " oranı ile " , tmp_i ,". soru:" , document[tmp_i])

max=0
i=0
for i in range(0,len(document)-2):
    tmp_cos=String_Similarity.cosine_similarity(tf_idf_vectors[i],tf_idf_vectors[query_index])
    if(max<tmp_cos):
        max=tmp_cos
        tmp_i=i

print(query, " sorusuna en benzer soru cosinüs benzerliğine göre" , tmp_cos , " oranı ile " , tmp_i ,". soru:" , document[tmp_i])


