from net import model_forward

def wrap_forward(enc, dec, start_id, eos_id):
    def wrapped(input_sit, force=False, sample=False, tu_target=None, loss_func=None, gen_length=13):
        return model_forward(enc,
                             dec,
                             input_sit,
                             start_id,
                             eos_id,
                             gen_length=gen_length,
                             force=force,
                             sample=sample,
                             tu_target=tu_target,
                             loss_func=loss_func)

    return wrapped
