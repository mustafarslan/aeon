#include "aeon/atlas.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

class AtlasTest : public ::testing::Test {
protected:
  std::filesystem::path test_path = "test_atlas.aeon";

  void TearDown() override {
    if (std::filesystem::exists(test_path)) {
      std::filesystem::remove(test_path);
    }
  }
};

TEST_F(AtlasTest, InsertAndRetrieve) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);

  uint64_t id = atlas.insert(0, vec, "Root Node");
  EXPECT_EQ(id, 0); // First node is 0
  EXPECT_EQ(atlas.size(), 1);
}

TEST_F(AtlasTest, NavigationSimple) {
  aeon::Atlas atlas(test_path);
  std::vector<float> zero(768, 0.0f);

  // Create Root
  atlas.insert(0, zero, "Root");

  // Create Child 1 (Close to target)
  std::vector<float> child1(768, 1.0f);
  atlas.insert(0, child1, "Child 1");

  // Create Child 2 (Far from target)
  std::vector<float> child2(768, -1.0f);
  atlas.insert(0, child2, "Child 2");

  // Query
  std::vector<float> query(768, 1.0f);
  auto path = atlas.navigate(query);

  // Should be Root -> Child 1
  // Should be Root -> Child 1 (or sorted by similarity)
  // Actually navigate sorts by similarity now.
  // Child 1 (1.0f) is closer to Query (1.0f) than Root (0.0f) or Child 2
  // (-1.0f). Root might be in list too. Let's just check the best one.
  ASSERT_GE(path.size(), 1);
  EXPECT_EQ(path[0].id, 1); // Child 1 has ID 1
}
