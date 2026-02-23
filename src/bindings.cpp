#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>

#include "engine.h"
#include "slb.h"

namespace nb = nanobind;

NB_MODULE(agentkv_core, m) {
    // -------------------------------------------------------------------------
    // 1. Expose Distance Metric Enum (must be before KVEngine which uses it)
    // -------------------------------------------------------------------------
    nb::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("COSINE", METRIC_COSINE)
        .value("L2", METRIC_L2)
        .value("INNER_PRODUCT", METRIC_INNER_PRODUCT);

    // -------------------------------------------------------------------------
    // 2. Expose KVEngine
    // -------------------------------------------------------------------------
    nb::class_<KVEngine>(m, "KVEngine")
        .def(nb::init<const std::string&, size_t, DistanceMetric>(), 
             nb::arg("path"), nb::arg("size_bytes"),
             nb::arg("metric") = METRIC_COSINE)
        
        // create_node: numpy array + optional text
        .def("create_node", [](KVEngine& self, uint64_t id, 
                                nb::ndarray<float, nb::ndim<1>> arr,
                                const std::string& text) {
            std::vector<float> vec(arr.data(), arr.data() + arr.shape(0));
            return self.create_node(id, vec, text);
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("text") = "",
           "Create a new node with vector embedding and optional text")
        
        // create_node: list<float> + optional text
        .def("create_node_list", [](KVEngine& self, uint64_t id, 
                                     const std::vector<float>& vec,
                                     const std::string& text) {
            return self.create_node(id, vec, text);
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("text") = "",
           "Create a new node with vector embedding (list) and optional text")
        
        .def("add_edge", &KVEngine::add_edge, 
             "Link two nodes (Directed)")

        // get_vector: zero-copy numpy view into mmap
        .def("get_vector", [](KVEngine& self, uint64_t node_offset) {
            auto [ptr, dim] = self.get_vector_raw(node_offset);
            if (!ptr) throw std::runtime_error("Node has no vector");
            size_t shape[1] = { dim };
            return nb::ndarray<nb::numpy, float>(
                ptr, 1, shape, nb::handle()
            );
        }, nb::rv_policy::reference_internal)

        // get_text: return Python str from mmap
        .def("get_text", [](KVEngine& self, uint64_t node_offset) -> std::string {
            auto [ptr, len] = self.get_text_raw(node_offset);
            if (!ptr) return "";
            return std::string(ptr, len);
        }, "Get the text content of a node")

        // search_knn: accept numpy query, return list of (offset, distance)
        .def("search_knn", [](KVEngine& self, 
                               nb::ndarray<float, nb::ndim<1>> query_arr,
                               int k, int ef_search) {
            return self.search_knn(query_arr.data(), 
                                   static_cast<uint32_t>(query_arr.shape(0)),
                                   k, ef_search);
        }, nb::arg("query"), nb::arg("k"), nb::arg("ef_search") = 50,
           nb::call_guard<nb::gil_scoped_release>(),
           "K-NN vector search. Returns list of (node_offset, distance)")

        // search_knn_filtered: same + metadata filters
        .def("search_knn_filtered", [](KVEngine& self,
                                        nb::ndarray<float, nb::ndim<1>> query_arr,
                                        int k, int ef_search,
                                        const std::vector<std::pair<std::string, std::string>>& filter_pairs) {
            std::vector<MetadataFilter> filters;
            filters.reserve(filter_pairs.size());
            for (auto& [key, val] : filter_pairs) {
                filters.push_back({key, val});
            }
            return self.search_knn_filtered(
                query_arr.data(),
                static_cast<uint32_t>(query_arr.shape(0)),
                k, ef_search, filters);
        }, nb::arg("query"), nb::arg("k"), nb::arg("ef_search") = 50,
           nb::arg("filters") = std::vector<std::pair<std::string, std::string>>{},
           nb::call_guard<nb::gil_scoped_release>(),
           "Filtered K-NN search. Filters: list of (key, value) pairs.")

        // insert: dynamic HNSW insertion (auto-wires bidirectional links)
        .def("insert", [](KVEngine& self, uint64_t id,
                           nb::ndarray<float, nb::ndim<1>> arr,
                           const std::string& text) {
            std::vector<float> vec(arr.data(), arr.data() + arr.shape(0));
            return self.insert(id, vec, text);
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("text") = "",
           nb::call_guard<nb::gil_scoped_release>(),
           "Insert a node with automatic HNSW index wiring")

        // insert_batch: bulk insert from 2D numpy array
        .def("insert_batch", [](KVEngine& self,
                                 nb::ndarray<uint64_t, nb::ndim<1>> ids_arr,
                                 nb::ndarray<float, nb::ndim<2>> data_arr,
                                 const std::vector<std::string>& texts) {
            uint32_t n = static_cast<uint32_t>(data_arr.shape(0));
            uint32_t dim = static_cast<uint32_t>(data_arr.shape(1));
            return self.insert_batch(ids_arr.data(), data_arr.data(),
                                     n, dim, texts);
        }, nb::arg("ids"), nb::arg("data"), nb::arg("texts"),
           nb::call_guard<nb::gil_scoped_release>(),
           "Batch insert N vectors. ids: (N,) uint64, data: (N,dim) float32, texts: list[str]")

        // delete / update
        .def("delete_node", &KVEngine::delete_node,
             "Tombstone a node (skipped in search and iteration)")
        .def("is_deleted", &KVEngine::is_deleted,
             "Check if a node has been deleted")
        .def("update_node", [](KVEngine& self, uint64_t old_offset, uint64_t new_id,
                                nb::ndarray<float, nb::ndim<1>> arr,
                                const std::string& text) {
            std::vector<float> vec(arr.data(), arr.data() + arr.shape(0));
            return self.update_node(old_offset, new_id, vec, text);
        }, nb::arg("old_offset"), nb::arg("new_id"),
           nb::arg("embedding"), nb::arg("text") = "",
           nb::call_guard<nb::gil_scoped_release>(),
           "Update = tombstone old + insert new. Returns new offset.")

        // metadata
        .def("set_metadata", &KVEngine::set_metadata,
             nb::arg("node_offset"), nb::arg("key"), nb::arg("value"),
             "Set a metadata key=value on a node")
        .def("get_metadata", &KVEngine::get_metadata,
             nb::arg("node_offset"), nb::arg("key"),
             "Get a metadata value by key")
        .def("get_all_metadata", &KVEngine::get_all_metadata,
             nb::arg("node_offset"),
             "Get all metadata as list of (key, value) pairs")

        // count / iteration
        .def("count", &KVEngine::count,
             "Number of live (non-deleted) nodes")
        .def("total_count", &KVEngine::total_count,
             "Total nodes ever created (including deleted)")
        .def("get_all_node_offsets", &KVEngine::get_all_node_offsets,
             nb::call_guard<nb::gil_scoped_release>(),
             "Get offsets of all live nodes")
        .def("get_metric", &KVEngine::get_metric,
             "Get the distance metric (0=cosine, 1=L2, 2=inner_product)")

        // HNSW management (kept for backward compat / manual use)
        .def("init_hnsw", &KVEngine::init_hnsw)
        .def("add_hnsw_link", &KVEngine::add_hnsw_link)

        // Crash recovery / validation
        .def("is_valid", &KVEngine::is_valid,
             "Check if the database header checksum is valid")
        .def("sync", &KVEngine::sync,
             "Force msync of the entire mmap to disk");

    // -------------------------------------------------------------------------
    // 3. Expose ContextManager (SLB)
    // -------------------------------------------------------------------------
    nb::class_<ContextManager>(m, "ContextManager")
        .def(nb::init<KVEngine*>(), nb::keep_alive<1, 2>())
        
        .def("observe", &ContextManager::observe_and_predict, 
             nb::call_guard<nb::gil_scoped_release>(),
             "Update the active context based on user focus")
        
        .def("get_context", &ContextManager::get_context_window, 
             "Get the current list of predicted node offsets");
}