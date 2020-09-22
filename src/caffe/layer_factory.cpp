// Make sure we include Python.h before any system header to avoid _POSIX_C_SOURCE redefinition
#include <string>
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#include "caffe/layers/python_layer.hpp"
#endif


namespace caffe {

// Get convolution layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConvolutionLayer(const LayerParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();
  ConvolutionParameter_Engine engine = conv_param.engine();

  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
  }

  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);


// Get pooling layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPoolingLayer(const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
  }

  if (engine == PoolingParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);


// Get LRN layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetLRNLayer(const LayerParameter& param) {
  LRNParameter_Engine engine = param.lrn_param().engine();

  if (engine == LRNParameter_Engine_DEFAULT) {
    engine = LRNParameter_Engine_CAFFE;
  }

  if (engine == LRNParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new LRNLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);


// Get relu layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetReLULayer(const LayerParameter& param) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);


// Get sigmoid layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSigmoidLayer(const LayerParameter& param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
  }
  
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SigmoidLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);


// Get softmax layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSoftmaxLayer(const LayerParameter& param) {
  SoftmaxParameter_Engine engine = param.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SoftmaxLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);


// Get tanh layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetTanHLayer(const LayerParameter& param) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
  }

  if (engine == TanHParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new TanHLayer<Dtype>(param));
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);


#ifdef WITH_PYTHON_LAYER
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPythonLayer(const LayerParameter& param) {
  Py_Initialize();
  try {
    bp::object module = bp::import(param.python_param().module().c_str());
    bp::object layer = module.attr(param.python_param().layer().c_str())(param);
    return bp::extract<shared_ptr<PythonLayer<Dtype> > >(layer)();
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer);
#endif

// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
