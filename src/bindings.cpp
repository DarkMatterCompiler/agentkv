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
        
        // Accept both list<float> and numpy ndarray
        .def("create_node", [](KVEngine& self, uint64_t id, nb::ndarray<float, nb::ndim<1>> arr) {
            // Convert ndarray to std::vector<float> for the C++ API
            std::vector<float> vec(arr.data(), arr.data() + arr.shape(0));
            return self.create_node(id, vec);
        }, "Create a new node with a vector embedding (accepts numpy array)")
        
        .def("create_node_list", &KVEngine::create_node, 
             "Create a new node with a vector embedding (accepts list)")
        
        .def("add_edge", &KVEngine::add_edge, 
             "Link two nodes (Directed)")

        .def("get_vector", [](KVEngine& self, uint64_t node_offset) {
            // ZERO-COPY MAGIC HAPPENS HERE
            auto [ptr, dim] = self.get_vector_raw(node_offset);
            
            if (!ptr) throw std::runtime_error("Node has no vector");

            // Create a numpy array view with dynamic shape.
            size_t shape[1] = { dim };
            return nb::ndarray<nb::numpy, float>(
                ptr, 
                1, 
                shape, 
                nb::handle() // The array doesn't own the data, KVEngine does
            );
        }, nb::rv_policy::reference_internal);


    // -------------------------------------------------------------------------
    // 2. Expose ContextManager (SLB)
    // -------------------------------------------------------------------------
    nb::class_<ContextManager>(m, "ContextManager")
        .def(nb::init<KVEngine*>(), nb::keep_alive<1, 2>()) // Keep Engine alive while Manager exists
        
        .def("observe", &ContextManager::observe_and_predict, 
             nb::call_guard<nb::gil_scoped_release>(), // Release GIL! Thread-safe.
             "Update the active context based on user focus")
        
        .def("get_context", &ContextManager::get_context_window, 
             "Get the current list of predicted node offsets");
}