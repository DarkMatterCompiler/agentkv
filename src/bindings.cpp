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
    // 1. Expose KVEngine
    // -------------------------------------------------------------------------
    nb::class_<KVEngine>(m, "KVEngine")
        .def(nb::init<const std::string&, size_t>(), 
             nb::arg("path"), nb::arg("size_bytes"))
        
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

        // insert: dynamic HNSW insertion (auto-wires bidirectional links)
        .def("insert", [](KVEngine& self, uint64_t id,
                           nb::ndarray<float, nb::ndim<1>> arr,
                           const std::string& text) {
            std::vector<float> vec(arr.data(), arr.data() + arr.shape(0));
            return self.insert(id, vec, text);
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("text") = "",
           nb::call_guard<nb::gil_scoped_release>(),
           "Insert a node with automatic HNSW index wiring")

        // HNSW management (kept for backward compat / manual use)
        .def("init_hnsw", &KVEngine::init_hnsw)
        .def("add_hnsw_link", &KVEngine::add_hnsw_link)

        // Crash recovery / validation
        .def("is_valid", &KVEngine::is_valid,
             "Check if the database header checksum is valid")
        .def("sync", &KVEngine::sync,
             "Force msync of the entire mmap to disk");


    // -------------------------------------------------------------------------
    // 2. Expose ContextManager (SLB)
    // -------------------------------------------------------------------------
    nb::class_<ContextManager>(m, "ContextManager")
        .def(nb::init<KVEngine*>(), nb::keep_alive<1, 2>())
        
        .def("observe", &ContextManager::observe_and_predict, 
             nb::call_guard<nb::gil_scoped_release>(),
             "Update the active context based on user focus")
        
        .def("get_context", &ContextManager::get_context_window, 
             "Get the current list of predicted node offsets");
}