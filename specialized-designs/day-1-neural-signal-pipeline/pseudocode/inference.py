# Simplified TensorRT inference (Python)
import tensorrt as trt
def infer_signal(model, preprocessed_data):
    with trt.Runtime() as runtime:
        engine = runtime.deserialize_cuda_engine(model)
        with engine.create_execution_context() as context:
            output = context.execute_v2(bindings=[preprocessed_data])
    return output  # Intent classification


