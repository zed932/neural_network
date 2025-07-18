import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np

def quantize_onnx_model(onnx_model_path: str, quantized_model_path: str):
    """
    Квантует ONNX модель динамически (INT8) и сохраняет результат
    :param onnx_model_path: Путь к исходной ONNX модели
    :param quantized_model_path: Путь для сохранения квантованной модели
    """
    # Динамическое квантование (веса INT8, активации float32)
    quantize_dynamic(
        input_model_path=onnx_model_path,
        output_model_path=quantized_model_path,
        weight_type=QuantType.QInt8,  # Квантование весов в INT8
        optimize_model=True,         # Дополнительная оптимизация графа
        op_types_to_quantize=['MatMul', 'Conv', 'Gemm', 'Add']  # Какие операторы квантовать
    )

def create_ort_session(model_path: str, use_quantized: bool = False):
    """
    Создает сессию ONNX Runtime для обычной или квантованной модели
    
    :param model_path: Путь к ONNX модели
    :param use_quantized: Использовать ли INT8-ускорители (для квантованных моделей)
    :return: InferenceSession
    """
    providers = ['CPUExecutionProvider']
    
    # Для квантованных моделей включаем оптимизации
    session_options = ort.SessionOptions()
    if use_quantized:
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    return ort.InferenceSession(
        model_path,
        providers=providers,
        sess_options=session_options
    )

def compare_models(onnx_path: str, quantized_path: str, input_sample: np.ndarray):
    """
    Сравнение оригинальной и квантованной моделей
    
    :param onnx_path: Путь к оригинальной ONNX модели
    :param quantized_path: Путь к квантованной модели
    :param input_sample: Пример входных данных (numpy array)
    """
    # Создаем сессии
    orig_sess = create_ort_session(onnx_path)
    quant_sess = create_ort_session(quantized_path, use_quantized=True)
    
    # Получаем имена входов (могут отличаться для разных моделей)
    input_name = orig_sess.get_inputs()[0].name
    
    # Замеряем производительность
    import time
    def benchmark(session):
        start = time.time()
        for _ in range(10):
            session.run(None, {input_name: input_sample})
        return (time.time() - start) / 10
    
    orig_time = benchmark(orig_sess)
    quant_time = benchmark(quant_sess)
    
    # Сравнение выходов
    orig_out = orig_sess.run(None, {input_name: input_sample})[0]
    quant_out = quant_sess.run(None, {input_name: input_sample})[0]
    
    print(f"Original model: {orig_time:.4f} sec/step")  # Оригинальная модель: время на шаг
    print(f"Quantized model: {quant_time:.4f} sec/step")  # Квантованная модель: время на шаг
    print(f"Speedup: {orig_time/quant_time:.1f}x")       # Ускорение в X раз
    print(f"Max output difference: {np.max(np.abs(orig_out - quant_out)):.4f}")  # Максимальное расхождение выходов