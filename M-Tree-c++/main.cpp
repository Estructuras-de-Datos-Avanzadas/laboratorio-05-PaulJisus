#include <iostream>

using namespace std;

#include "mtree.cpp"

int main()
{
    Mtree mtree{5};
    int N;
    cin>>N;
    float i=0;
    while(N--){
        string a;
        cin>>a;
        int x, y;
        cin>>x>>y;
        float features[2] = {i+1, i+2};
        Embedding emb = {features, 2, a};
        mtree.addObject(emb);
        i++;
    }
    printTree(mtree);
    float features[2] = {2,3};
    Embedding embedding = {features, 2, "Peru"};
    vector<Embedding> results = ConsultaRango(mtree.root, embedding, 3);
    cout<<"\n\nConsultaRango: "<<endl;
    for(Embedding i : results){
        printEmbedding(i);
    }
    cout<<"\n\nConsultaRango Radio: "<<endl;
    set<string> indexResults = diversedConsultaRango(mtree, embedding, 10, 4);

    for (auto embeddingResult : indexResults)
        cout << embeddingResult << "    ";
}
