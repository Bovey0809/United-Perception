import torch
import torchvision
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization

def test_nnie_export():
    # 创建基础模型
    model_to_quantize = torchvision.models.resnet18(pretrained=False)
    model_to_quantize.train()
    
    # 创建输入
    dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
    
    # 配置NNIE量化参数
    extra_qconfig_dict = {
        'w_observer': 'MinMaxObserver',
        'a_observer': 'EMAMinMaxObserver',
        'w_fakequantize': 'NNIEFakeQuantize',
        'a_fakequantize': 'NNIEFakeQuantize',
    }
    prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
    
    # 准备量化模型
    model_prepared = prepare_by_platform(
        model_to_quantize, 
        BackendType.NNIE,
        prepare_custom_config_dict
    )
    
    # 启用校准并收集统计信息
    enable_calibration(model_prepared)
    model_prepared(dummy_input)
    
    # 启用量化
    enable_quantization(model_prepared)
    
    # 运行一次前向传播
    with torch.no_grad():
        _ = model_prepared(dummy_input)
    
    # 设置为评估模式
    model_prepared.eval()
    
    # 转换并部署模型
    convert_deploy(
        model_prepared,
        BackendType.NNIE,
        {'x': [1, 3, 224, 224]},
        dummy_input=dummy_input,
        model_name='resnet18_nnie'
    )
    print("NNIE model export successful!")
    
    # 验证生成的ONNX模型
    import onnx
    onnx_model = onnx.load("resnet18_nnie.onnx")
    print("\nONNX model nodes:")
    for node in onnx_model.graph.node:
        if "QuantizeLinear" in node.op_type:
            print(f"Op type: {node.op_type}")
            print(f"Inputs: {node.input}")
            print(f"Outputs: {node.output}")
            print("Attributes:", [f"{attr.name}: {attr.f if attr.name.endswith('_f') else attr.i}" 
                                for attr in node.attribute])
            print()
            
if __name__ == "__main__":
    test_nnie_export() 