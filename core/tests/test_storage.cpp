#include "aeon/storage.hpp"
#include <filesystem>
#include <gtest/gtest.h>

class StorageTest : public ::testing::Test {
protected:
  std::filesystem::path test_path = "test_storage.aeon";

  void TearDown() override {
    if (std::filesystem::exists(test_path)) {
      std::filesystem::remove(test_path);
    }
  }
};

TEST_F(StorageTest, CreateAndOpen) {
  aeon::storage::MemoryFile file;
  auto result = file.open(test_path, 100);
  ASSERT_TRUE(result.has_value());

  auto *header = file.get_header();
  ASSERT_NE(header, nullptr);
  EXPECT_EQ(header->magic, aeon::ATLAS_MAGIC);
  EXPECT_EQ(header->capacity, 100);
}

TEST_F(StorageTest, Persistence) {
  {
    aeon::storage::MemoryFile file;
    file.open(test_path, 10);
    auto *node = file.get_node(0);
    node->id = 12345;
    node->centroid[0] = 3.14f;
  } // Close

  {
    aeon::storage::MemoryFile file;
    file.open(test_path); // Reopen
    auto *node = file.get_node(0);
    EXPECT_EQ(node->id, 12345);
    EXPECT_FLOAT_EQ(node->centroid[0], 3.14f);
  }
}

TEST_F(StorageTest, Grow) {
  aeon::storage::MemoryFile file;
  file.open(test_path, 10);
  EXPECT_EQ(file.get_header()->capacity, 10);

  // Write something at the end
  file.get_node(9)->id = 999;

  auto res = file.grow(20);
  ASSERT_TRUE(res.has_value());
  EXPECT_EQ(file.get_header()->capacity, 20);

  // Verify old data is still there
  EXPECT_EQ(file.get_node(9)->id, 999);

  // Access new space
  file.get_node(15)->id = 888;
  EXPECT_EQ(file.get_node(15)->id, 888);
}
