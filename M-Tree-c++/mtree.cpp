#include <queue>
#include "mtree.h"

void Mtree::addObject(Embedding embedding) {
    size++;
    root->addObject(embedding);
}

Mtree::Mtree(int maxNodeSize_) {
    maxNodeSize = maxNodeSize_;
    this->root = new Node(true, this);
}

vector<Embedding> ConsultaRango(Node * N, Embedding embedding, float range){
    vector<Embedding> results;
    Entry * Op = N->parentEntry;
    if (!N->isRoot()) {
        if (!N->isLeaf) {
            for (Entry Or : N->entries) {
                if (abs( distance(*(Op->embedding), embedding) - Or.distanceToParent ) <= range + Or.radius) {
                    if (distance(*(Or.embedding), embedding) <= range + Or.radius) {
                        vector<Embedding> tempResults = ConsultaRango(Or.subTree, embedding, range);
                        results.insert(results.end(), tempResults.begin(), tempResults.end());
                    }
                }
            }
        } else {
            for (Entry Oj : N->entries) {
                if (abs( distance(*(Op->embedding), embedding) - Oj.distanceToParent ) <= range) {
                    if (distance(*(Oj.embedding), embedding) <= range) {
                        results.push_back(*(Oj.embedding));
                    }
                }
            }
        }
    } else {
        if (!N->isLeaf) {
            for (auto Or : N->entries) {
                if (distance(*(Or.embedding), embedding) <= range + Or.radius) {
                    vector<Embedding> tempResults = ConsultaRango(Or.subTree, embedding, range);
                    results.insert(results.end(), tempResults.begin(), tempResults.end());
                }
            }
        } else {
            for (auto Oj : N->entries) {
                if (distance(*(Oj.embedding), embedding) <= range) {
                    results.push_back(*(Oj.embedding));
                }
            }
        }
    }
    return results;
}

set<string> diversedConsultaRango(Mtree mtree, Embedding embedding, float range, int k){
    queue<Node*> nodeQueue;
    set<string> results;
    nodeQueue.push(mtree.root);
    while(!nodeQueue.empty() && results.size() < k) {
        Node * node = nodeQueue.front();
        Entry * Op = node->parentEntry;
        nodeQueue.pop();

        if (node == NULL) continue;
        if (!node->isRoot()) {
            if (!node->isLeaf) {
                for (auto Or : node->entries) {
                    if (abs( distance(*(Op->embedding), embedding) - Or.distanceToParent ) <= range + Or.radius) {
                        float distanceToQuery = distance(*(Or.embedding), embedding);
                        if (distanceToQuery <= range + Or.radius) {
                            nodeQueue.push(Or.subTree);
                            if (distanceToQuery <= range) {
                                results.insert(Or.embedding->id);
                                if (results.size() >=k)
                                    return results;
                            }
                        }
                    }
                }
            } else {
                for (Entry Oj : node->entries) {
                    if (abs( distance(*(Op->embedding), embedding) - Oj.distanceToParent ) <= range) {
                        if (distance(*(Oj.embedding), embedding) <= range) {
                            results.insert(Oj.embedding->id);
                        }
                    }
                }
            }
        } else {
            if (!node->isLeaf) {
                for (auto Or : node->entries) {
                    float distanceToQuery = distance(*(Or.embedding), embedding);
                    if (distanceToQuery <= range + Or.radius) {
                        nodeQueue.push(Or.subTree);
                        if (distanceToQuery <= range) {
                            results.insert(Or.embedding->id);
                            if (results.size() >=k)
                                return results;
                        }
                    }
                }
            } else {
                for (Entry Oj : node->entries) {
                    if (distance(*(Oj.embedding), embedding) <= range) {
                        results.insert(Oj.embedding->id);
                    }
                }
            }
        }
    }
    return results;
}

float distance(Embedding x, Embedding y) {
    assert (x.len == y.len);
    float dist = 0;
    for (int i =0 ;i < x.len; i++) {
        dist += pow(x.features[i] - y.features[i],2);
    }
    return sqrt(dist);
}

void promote(vector<Entry> allEntries, Embedding& routingObject1, Embedding& routingObject2) {
    float maxDistance = 0;
    for (int i=0; i < allEntries.size() ;i++) {
        for (int j=i; j< allEntries.size(); j++) {
            float dist = distance(*(allEntries.at(i).embedding), *(allEntries.at(j).embedding));
            if (dist>maxDistance) {
                routingObject1 = *(allEntries.at(i).embedding);
                routingObject2 = *(allEntries.at(j).embedding);
                maxDistance = dist;
            }
        }
    }
}

void partition(vector<Entry> allEntries, vector<Entry>& entries1, vector<Entry>& entries2, const Embedding& routingObject1, const Embedding& routingObject2){
    for (Entry entry : allEntries) {
        if (distance(*(entry.embedding), routingObject1) <= distance(*(entry.embedding), routingObject2) )
            entries1.push_back(entry);
        else
            entries2.push_back(entry);
    }
}

void printEmbedding(Embedding embedding) {
    cout << "[ ";
    for (int i =0;i < embedding.len;i++)
        cout << embedding.features[i] << " ";
    cout << " ]" << endl;
}

void printTree(Mtree mtree) {
    queue<Node*> nodeQueue;

    cout << "Root: " << mtree.root << endl;
    nodeQueue.push(mtree.root);
    while(!nodeQueue.empty()) {
        Node * node = nodeQueue.front();
        cout << endl;
        if (node->isLeaf)
            cout << "Address leaf nodo " << node << endl;
        else
            cout << "Address internal nodo " << node << endl;
        nodeQueue.pop();

        if (node == NULL) continue;
        if (node->parentEntry) {
            cout << "parent: " << node->parentEntry->embedding->id << "(" << node->parentEntry->radius << "," << node->parentEntry->distanceToParent << ")" << endl;
        }
        else {
            cout<< "no parent " <<endl;
        }
        for (Entry entry : node->entries) {
            cout << entry.embedding->id << " (" << entry.radius << "," << entry.distanceToParent << ")" << "   -    ";
            if (!node->isLeaf)
                nodeQueue.push(entry.subTree);
        }
        cout<<endl;
        cout<<endl;
    }
}
