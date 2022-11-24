#ifndef _MTREE_HPP_
#define _MTREE_HPP_

#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <limits>
#include <assert.h>
#include <string>
using namespace std;

class Embedding;
class Node;
class Entry;
class Mtree;

float distance(Embedding x, Embedding y);
void promote(vector<Entry> allEntries, Embedding& routingObject1, Embedding& routingObject2);
void partition(vector<Entry> allEntries, vector<Entry>& entries1, vector<Entry>& entries2, const Embedding& routingObject1, const Embedding& routingObject2);
vector<Embedding> queryRange(Node * node, Embedding embedding, float range);
set<string> diversedQueryRange(Mtree mtree, Embedding embedding, float range, int k);
void printEmbedding(Embedding embedding);
void printTree(Mtree mtree);

class Embedding {
public:
    float * features;
    int len;
    string id;

    Embedding(float * features_, int len_, string id_) {
        features = new float[len_]();
        len = len_;
        for (int i =0 ; i< len;i++)
            features[i] = features_[i];
        id = id_;
    }

    Embedding() {}
};

class Mtree {
public:
    int maxNodeSize = 5;
    int size = 0;
    Node * root;
    friend class Node;
public:
    Mtree(int maxNodeSize_);

    void addObject(Embedding embedding);

};

class Entry{
public:
    Embedding * embedding;
    float distanceToParent;
    float radius;
    Node * subTree;
    friend class MTree;
    friend class Node;
public:
        Entry(Embedding * embedding_, float distanceToParent_, float radius_, Node * subTree_) :
        subTree(subTree_) {
            embedding = new Embedding();
            *embedding = *embedding_;
            distanceToParent = distanceToParent_;
            radius = radius_;
            subTree = subTree_;
        }
    bool operator < (const Entry &other) const {
        for (int i=0; i < embedding->len; i++)
            if (embedding->features[i] != other.embedding->features[i])
                return embedding->features[i] < other.embedding->features[i];
        return true;
    }

    bool operator == (const Entry &other) {
        for (int i=0; i < embedding->len; i++)
            if (embedding->features[i] != other.embedding->features[i])
                return false;
        if (subTree != other.subTree)
            return false;

        return true;
    }
};

class Node {
public:
    bool isLeaf = true;
    Mtree * mtree;
    Node * parentNode;
    vector<Entry> entries;
    Entry * parentEntry;

    Node(bool isLeaf_, Mtree * mtree_, Node * parentNode_, vector<Entry> entries_, Entry * parentEntry_):
        isLeaf(isLeaf_), mtree(mtree_), parentNode(parentNode_), entries(entries_), parentEntry(parentEntry_) {}

    Node(bool isLeaf_, Mtree * mtree_) {
        isLeaf = isLeaf_;
        mtree = mtree_;
    }

    bool isFull() {
        return entries.size() == mtree->maxNodeSize;
    }

    bool isRoot() {
        return (this == this->mtree->root);
    }

    void setEntriesAndParentEntry(vector<Entry> entries, Entry* parentEntry) {
        this->entries = entries;
        this->parentEntry = parentEntry;
        this->parentEntry->radius = updateRadius(parentEntry->embedding);
        this->updateEntryDistanceToParent();
        if (!this->isLeaf)
            for (auto& entry : this->entries)
                entry.subTree->parentNode = this;
    }

    float updateRadius(Embedding* embedding) {
        float maxRadius = 0;
        if (this->isLeaf) {
            for (auto entry : this->entries) {
                float radius = distance(*embedding, *(entry.embedding));
                if (radius > maxRadius)
                    maxRadius = radius;
            }
        } else {
            for (auto entry : this->entries) {
                float radius = distance(*embedding, *(entry.embedding)) + entry.radius;
                if (radius > maxRadius)
                    maxRadius = radius;
            }
        }
        return maxRadius;
    }

    void updateEntryDistanceToParent() {
        if (this->parentEntry != NULL) {
            for (auto& entry : this->entries) {
                entry.distanceToParent = distance(*(entry.embedding), *(this->parentEntry->embedding));
            }
        }
    }

    void addObject(Embedding embedding) {
        if (this->isLeaf) {
            this->addObjectToLeaf(embedding);
        } else {
            this->addObjectToInner(embedding);
        }
    }

    void addObjectToLeaf(Embedding embedding) {
        float distanceToParent = -1;
        if (this->parentEntry) {
            if (this->parentEntry->embedding) {
                distanceToParent = distance(embedding, *(this->parentEntry->embedding));
            }
        }

        Entry newEntry = {&embedding, distanceToParent, -1, NULL};
        if (!this->isFull()) {
            this->entries.push_back(newEntry);
        } else {
            this->split(newEntry);
        }
        assert(this->isRoot() || this->parentNode);
    }

    void addObjectToInner(Embedding embedding) {
        Entry* bestEntry;
        float minDistance = numeric_limits<float>::max();
        vector<float> distances;
        bool requireNoRadiusIncrease = false;
        for (vector<Entry>::iterator entry = this->entries.begin(); entry < this->entries.end(); entry++) {
            float distanceToObject = distance(*(entry->embedding), embedding);
            if (distanceToObject <= entry->radius) requireNoRadiusIncrease = true;
            distances.push_back(distanceToObject);
        }
        if (requireNoRadiusIncrease) {
            int i = 0;
            for (vector<Entry>::iterator entry = this->entries.begin(); entry < this->entries.end(); entry++) {
                float distanceToObject = distances.at(i);
                if (distanceToObject < minDistance) {
                    minDistance = distanceToObject;
                    bestEntry = &(*entry);
                }
                i++;
            }
        } else {
            int bestIndex = 0;
            int i = 0;
            for (vector<Entry>::iterator entry = this->entries.begin(); entry < this->entries.end(); entry++) {
                float distanceToObject = distances.at(i) - entry->radius;
                if (distanceToObject < minDistance) {
                    minDistance = distanceToObject;
                    bestEntry = &(*entry);
                    bestIndex = i;
                }
                i++;
            }
            bestEntry->radius = distances.at(bestIndex);
        }
        bestEntry->subTree->addObject(embedding);

        assert(this->isRoot() || this->parentNode);
    }

    void split(Entry newEntry) {
        assert(this->isFull());
        Mtree* mtree = this->mtree;
        Node* newNode = new Node (this->isLeaf, mtree);
        vector<Entry> allEntries = this->entries;
        allEntries.push_back(newEntry);
        Embedding routingObject1, routingObject2;
        promote(allEntries, routingObject1, routingObject2);

        vector<Entry> entries1, entries2;
        partition(allEntries, entries1, entries2, routingObject1, routingObject2);
        Entry* oldParentEntry = this->parentEntry;
        Entry* existNodeEntry = new Entry (&routingObject1, -1, -1, this);
        this->setEntriesAndParentEntry(entries1, existNodeEntry);

        Entry * newNodeEntry = new Entry (&routingObject2, -1, -1, newNode);
        newNode->setEntriesAndParentEntry(entries2, newNodeEntry);

        if (this->isRoot()) {
            Node * newRootNode = new Node (false, mtree);
            this->parentNode = newRootNode;
            newRootNode->entries.push_back(*existNodeEntry);
            newNode->parentNode = newRootNode;
            newRootNode->entries.push_back(*newNodeEntry);
            mtree->root = newRootNode;

        } else {
            Node * parentNode = this->parentNode;
            if (!parentNode->isRoot()) {
                existNodeEntry->distanceToParent = distance(*(existNodeEntry->embedding), *(parentNode->parentEntry->embedding));
                newNodeEntry->distanceToParent = distance(*(newNodeEntry->embedding), *(parentNode->parentEntry->embedding));
            }
            for (vector<Entry>::iterator it = parentNode->entries.begin(); it < parentNode->entries.end(); it++) {
                if (*it == *oldParentEntry) {
                    parentNode->entries.erase(it);
                    break;
                }
            }
            parentNode->entries.push_back(*existNodeEntry);

            if (parentNode->isFull()) {
                parentNode->split(*newNodeEntry);
            }
            else {
                parentNode->entries.push_back(*newNodeEntry);
                newNode->parentNode = parentNode;
            }
        }
        assert(this->isRoot() || this->parentNode);
        assert(newNode->isRoot() || newNode->parentNode);
    }

};

#endif
