

def model_summary(model, input_shape):
    try:
        from torchsummary import summary
        
        summary(model, input_shape)
    except:
        print("Could not print summary")


def model_size(model):
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.nelement()
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    buffer_count = 0
    for buff in model.buffers():
        buffer_count += buff.nelement()
        buffer_size += buff.nelement() * buff.element_size()
    
    return {
        "param": {
            "count": param_count,
            "size": param_size / 1024**2,
            "unit": "MB"
        },
        "buffer": {
            "count": buffer_count,
            "size": buffer_size / 1024**2,
            "unit": "MB"
        }
    }
