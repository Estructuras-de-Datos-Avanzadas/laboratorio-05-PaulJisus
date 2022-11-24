#undef NDEBUG

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
#include <cassert>
#include "mtree.h"
#include "functions.h"
#include "tests/fixture.h"


#define assertEqual(A, B)             assert(A == B);
#define assertLessEqual(A, B)         assert(A <= B);
#define assertIn(ELEM, CONTAINER)     assert(CONTAINER.find(ELEM) != CONTAINER.end());
#define assertNotIn(ELEM, CONTAINER)  assert(CONTAINER.find(ELEM) == CONTAINER.end());

using namespace std;


typedef vector<int> Data;
typedef set<Data> DataSet;
typedef mt::functions::cached_distance_function<Data, mt::functions::euclidean_distance> CachedDistanceFunction;
typedef pair<Data, Data>(*PromotionFunction)(const DataSet&, CachedDistanceFunction&);

PromotionFunction nonRandomPromotion =
	[](const DataSet& dataSet, CachedDistanceFunction&) -> pair<Data, Data> {
		vector<Data> dataObjects(dataSet.begin(), dataSet.end());
		sort(dataObjects.begin(), dataObjects.end());
		return {dataObjects.front(), dataObjects.back()};
	};


typedef mt::mtree<
		Data,
		mt::functions::euclidean_distance,
		mt::functions::split_function<
				PromotionFunction,
				mt::functions::balanced_partition
			>
	>
	MTree;


class MTreeTest : public MTree {
private:
	struct OnExit {
		MTreeTest* mt;
		OnExit(MTreeTest* mt) : mt(mt) {}
		~OnExit() { mt->_check(); }
	};

public:

	using MTree::distance_function;

	MTreeTest()
		: MTree(2, -1,
				distance_function_type(),
				split_function_type(nonRandomPromotion)
			)
		{}

	void add(const Data& data) {
		OnExit onExit(this);
		return MTree::add(data);
	}

	bool remove(const Data& data) {
		OnExit onExit(this);
		return MTree::remove(data);
	}
};



class Test {
public:
	void testEmpty() {
		_checkNearestByRange({1, 2, 3}, 4);
		_checkNearestByLimit({1, 2, 3}, 4);
	}

	void test01() { _test("f01"); }
	void test02() { _test("f02"); }

	void testRemoveNonExisting() {
		assert(!mtree.remove({99, 77}));
		mtree.add({4, 44});
		assert(!mtree.remove({99, 77}));
		mtree.add({95, 43});
		assert(!mtree.remove({99, 77}));
		mtree.add({76, 21});
		assert(!mtree.remove({99, 77}));
		mtree.add({64, 53});
		assert(!mtree.remove({99, 77}));
		mtree.add({47, 3});
		assert(!mtree.remove({99, 77}));
		mtree.add({26, 11});
		assert(!mtree.remove({99, 77}));
	}

	void testNotRandom() {
		const string fixtureName = "fNotRandom";
		string fixtureFileName = Fixture::path(fixtureName);
		if(!ifstream(fixtureFileName)) {
			cout << "\tskipping..." << endl;
			return;
		}
		_test(fixtureName.c_str());
	}
	}

private:
	typedef vector<MTreeTest::query::result_item> ResultsVector;

	MTreeTest mtree;
	set<Data> allData;

	void _test(const char* fixtureName) {
		Fixture fixture = Fixture::load(fixtureName);
		_testFixture(fixture);
	}

	void _testFixture(const Fixture& fixture) {
		for(vector<Fixture::Action>::const_iterator i = fixture.actions.begin(); i != fixture.actions.end(); ++i) {
			switch(i->cmd) {
			case 'A':
				allData.insert(i->data);
				mtree.add(i->data);
				break;
			case 'R': {
				allData.erase(i->data);
				bool removed = mtree.remove(i->data);
				assert(removed);
				break;
			}
			default:
				cerr << i->cmd << endl;
				assert(false);
				break;
			}

			_checkNearestByRange(i->queryData, i->radius);
			_checkNearestByLimit(i->queryData, i->limit);
		}
	}

	void _checkNearestByRange(const Data& queryData, double radius) const {
		ResultsVector results;
		set<Data> strippedResults;
		MTreeTest::query query = mtree.get_nearest_by_range(queryData, radius);

		for(MTreeTest::query::iterator i = query.begin(); i != query.end(); ++i) {
			MTreeTest::query::value_type r = *i;
			results.push_back(r);
			strippedResults.insert(r.data);
		}

		double previousDistance = 0;

		for(ResultsVector::iterator i = results.begin(); i != results.end(); i++) {

			assertLessEqual(previousDistance, i->distance);
			previousDistance = i->distance;
			assertIn(i->data, allData);
			assertLessEqual(i->distance, radius);
			assertEqual(mtree.distance_function(i->data, queryData), i->distance);
		}

		for(set<Data>::const_iterator data = allData.begin(); data != allData.end(); ++data) {
			double distance = mtree.distance_function(*data, queryData);
			if(distance <= radius) {
				assertIn(*data, strippedResults);
			} else {
				assertNotIn(*data, strippedResults);
			}
		}
	}


	void _checkNearestByLimit(const Data& queryData, unsigned int limit) const {
		ResultsVector results;
		set<Data> strippedResults;
		MTreeTest::query query = mtree.get_nearest_by_limit(queryData, limit);
		for(MTreeTest::query::iterator i = query.begin(); i != query.end(); i++) {
			MTreeTest::query::result_item r = *i;
			results.push_back(r);
			strippedResults.insert(r.data);
		}

		if(limit <= allData.size()) {
			assertEqual(limit, results.size());
		} else {
			assertEqual(allData.size(), results.size());
		}

		double farthest = 0.0;
		double previousDistance = 0.0;
		for(ResultsVector::iterator i = results.begin(); i != results.end(); ++i) {
			assertLessEqual(previousDistance, i->distance);
			previousDistance = i->distance;
			assertIn(i->data, allData);
			assertEqual(1, count(strippedResults.begin(), strippedResults.end(), i->data));
			double distance = mtree.distance_function(i->data, queryData);
			assertEqual(distance, i->distance);
			farthest = max(farthest, distance);
		}

		for(set<Data>::iterator pData = allData.begin(); pData != allData.end(); ++pData) {
			double distance = mtree.distance_function(*pData, queryData);
			if(distance < farthest) {
				assertIn(*pData, strippedResults);
			} else if(distance > farthest) {
				assertNotIn(*pData, strippedResults);
			} else {
			}
		}
	}
};

int main() {
#define RUN_TEST(T)   cout << "Running " #T "..." << endl; Test().T()
	RUN_TEST(testEmpty);
	RUN_TEST(test01);
	RUN_TEST(test02);
	RUN_TEST(testNotRandom);
#undef RUN_TEST
	cout << "DONE" << endl;
	return 0;
}
