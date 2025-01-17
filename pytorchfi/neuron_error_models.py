"""pytorchfi.error_models provides different error models out-of-the-box for use."""

import logging
import random
import struct

import torch

from pytorchfi import core
from pytorchfi.util import random_value

# Helper Functions

logger=logging.getLogger("pytorchfi") 
logger.setLevel(logging.DEBUG) 

def random_batch_element(pfi: core.FaultInjection):
    return random.randint(0, pfi.batch_size - 1)


def random_neuron_location(pfi: core.FaultInjection, layer: int = -1):
    if layer == -1:
        layer = random.randint(0, pfi.get_total_layers() - 1)

    dim = pfi.get_layer_dim(layer)
    shape = pfi.get_layer_shape(layer)

    dim1_shape = shape[1]
    dim1_rand = random.randint(0, dim1_shape - 1)
    if dim > 2:
        dim2_shape = shape[2]
        dim2_rand = random.randint(0, dim2_shape - 1)
    else:
        dim2_rand = None
    if dim > 3:
        dim3_shape = shape[3]
        dim3_rand = random.randint(0, dim3_shape - 1)
    else:
        dim3_rand = None

    return (layer, dim1_rand, dim2_rand, dim3_rand)


# Neuron Perturbation Models

# single random neuron error in single batch element
def random_neuron_inj(pfi: core.FaultInjection, min_val: int = -1, max_val: int = 1):
    b = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi.declare_neuron_fault_injection(
        batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
    )


# single random neuron error in each batch element.
def random_neuron_inj_batched(
    pfi: core.FaultInjection,
    min_val: int = -1,
    max_val: int = 1,
    rand_loc: bool = True,
    rand_val: bool = True,
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for _ in range(6))

    if not rand_loc:
        (layer, C, H, W) = random_neuron_location(pfi)
    if not rand_val:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi.batch_size):
        if rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi)
        if rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in single batch element
def random_inj_per_layer(pfi: core.FaultInjection, min_val: int = -1, max_val: int = 1):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    b = random_batch_element(pfi)
    for i in range(pfi.get_total_layers()):
        (layer, C, H, W) = random_neuron_location(pfi, layer=i)
        batch.append(b)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in each batch element
def random_inj_per_layer_batched(
    pfi: core.FaultInjection,
    min_val: int = -1,
    max_val: int = 1,
    rand_loc: bool = True,
    rand_val: bool = True,
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    for i in range(pfi.get_total_layers()):
        if not rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi, layer=i)
        if not rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi.batch_size):
            if rand_loc:
                (layer, C, H, W) = random_neuron_location(pfi, layer=i)
            if rand_val:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            layer_num.append(layer)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


class single_bit_flip_func(core.FaultInjection):
    def __init__(self, model, batch_size, input_shape=None, layer_types=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, layer_types=layer_types, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.bits = kwargs.get("bits", 8)
        self.layer_ranges = []

    def set_conv_max(self, data):
        self.layer_ranges = data

    def reset_conv_max(self, data):
        self.layer_ranges = []

    def get_conv_max(self, layer):
        return self.layer_ranges[layer]
    
    @staticmethod
    def _float_to_hex(f):
        h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
        return h[2:len(h)]

    @staticmethod
    def _hex_to_float(h):
        return float(struct.unpack(">f",struct.pack(">I",int(h,16)))[0])

    @staticmethod
    def _int_to_float(h):
        return float(struct.unpack(">f",struct.pack(">I",h))[0])
    
    @staticmethod
    def _max_num_bits(data):
        dtype=data.dtype
        if (dtype==torch.float32) or (dtype==torch.int32) or (dtype==torch.int):
            return(32)
        elif (dtype==torch.half) or (dtype==torch.float16) or (dtype==torch.int16) or (dtype==torch.short):
            return(16)
        elif (dtype==torch.int8) or (dtype==torch.uint8):
            return(8)
        else:
            return(32)
        

    
    # def _bit_flip_value(self,orig_value, bit_pos):
    #     save_type = orig_value.dtype
    #     orig_data=float(orig_value)
    #     injmask=2**bit_pos
    #     data_32bit=int(self._float_to_hex(orig_data),16)
    #     corrupt_32bit=data_32bit ^ int(injmask)
    #     corrupt_val=self._int_to_float(corrupt_32bit)
    #     return torch.tensor(corrupt_val, dtype=save_type)
    
    def _bit_flip_value(self,orig_values, bit_pos):
        # Convert tensor to float values
        orig_data = orig_values.float()

        # Generate injection mask for bit flip
        injmask = 2 ** bit_pos

        data_32bit = orig_data.view(torch.int32)
        corrupt_32bit = torch.bitwise_xor(data_32bit,injmask.type(torch.int32))
        corrupt_val = corrupt_32bit.view(torch.float)
        return corrupt_val

    @staticmethod
    def _twos_comp(val, bits):
        if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
        return val

    def _twos_comp_shifted(self, val, nbits):
        return (1 << nbits) + val if val < 0 else self._twos_comp(val, nbits)

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logger.info(f"Original Value: {orig_value}")

        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logger.info(f"Quantum: {quantum}")
        logger.info(f"Twos Couple: {twos_comple}")

        # binary representation
        bits = bin(twos_comple)[2:]
        logger.info(f"Bits: {bits}")

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        if len(bits) != total_bits:
            raise AssertionError
        logger.info(f"Sign extend bits {bits}")

        # flip a bit
        # use MSB -> LSB indexing
        if bit_pos >= total_bits:
            raise AssertionError

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logger.info(f"New bits: {bits_str_new}")

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logger.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        if not bits_str_new.isdigit():
            raise AssertionError
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logger.info(f"Out: {out}")

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logger.info(f"New Value: {new_value}")
        
        return torch.tensor(new_value, dtype=save_type)

    def single_bit_flip_signed_across_batch(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        range_max = self.get_conv_max(self.current_layer)
        logger.info(f"Current layer: {self.current_layer}")
        logger.info(f"Range_max: {range_max}")

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                if(i<output.shape[0]):
                    self.assert_injection_bounds(index=i)
                    prev_value = output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                        self.corrupt_dim[1][i]
                    ][self.corrupt_dim[2][i]]

                    rand_bit = random.randint(0, self.bits - 1)
                    logger.info(f"Random Bit: {rand_bit}")
                    new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                    output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                        self.corrupt_dim[1][i]
                    ][self.corrupt_dim[2][i]] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim[0]][
                    self.corrupt_dim[1]
                ][self.corrupt_dim[2]]

                rand_bit = random.randint(0, self.bits - 1)
                logger.info(f"Random Bit: {rand_bit}")
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
                    self.corrupt_dim[2]
                ] = new_value

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()

    def single_bit_flip_across_batch(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        bit_flip_pos = self.get_conv_max(0)
        logger.info(f"Current layer: {self.current_layer}")
        #logger.info(f"Range_max: {range_max}")
        
        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                #print(self.output_size[self.current_layer])
                if(i<output.shape[0]):
                    self.assert_injection_bounds(index=i)
                    prev_value = output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                        self.corrupt_dim[1][i]
                    ][self.corrupt_dim[2][i]]

                    #rand_bit = random.randint(0, self._max_num_bits(prev_value) - 1)
                    # rand_bit = random.randint(0, bit_flip_pos)
                    rand_bit = bit_flip_pos
                    logger.info(f"Random Bit: {rand_bit}")
                    new_value = self._bit_flip_value(prev_value, rand_bit)

                    output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                        self.corrupt_dim[1][i]
                    ][self.corrupt_dim[2][i]] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim[0]][
                    self.corrupt_dim[1]
                ][self.corrupt_dim[2]]

                # rand_bit = random.randint(0, self._max_num_bits(prev_value) - 1)
                # rand_bit = random.randint(0, bit_flip_pos)
                rand_bit = bit_flip_pos
                logger.info(f"Random Bit: {rand_bit}")
                new_value = self._bit_flip_value(prev_value, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
                    self.corrupt_dim[2]
                ] = new_value

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()

    def single_bit_flip_across_batch_tensor(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        bit_flip_pos = self.get_conv_max(0)
        logger.info(f"Current layer: {self.current_layer}")
        #logger.info(f"Range_max: {range_max}")
        
        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )                  
            if(len(inj_list)>0):                 
                dim=len(list(output.size()))
                indices_dim0 = torch.tensor(self.corrupt_batch) # batch
                indices_dim1 = torch.tensor(self.corrupt_dim[0][inj_list[0]:inj_list[0]+len(inj_list)]) # channel
                if(dim>2):
                    indices_dim2 = torch.tensor(self.corrupt_dim[1][inj_list[0]:inj_list[0]+len(inj_list)]) # row
                    indices_dim3 = torch.tensor(self.corrupt_dim[2][inj_list[0]:inj_list[0]+len(inj_list)]) #colum

                for i in range(output.shape[0]):                    
                    #self.assert_injection_bounds(index=i)
                    if dim>2:
                        prev_value = output[i, indices_dim1, indices_dim2, indices_dim3]
                    else:
                        prev_value = output[i, indices_dim1]

                    rand_bit = torch.tensor([bit_flip_pos],device=output.device.type)

                    logger.info(f"Random Bit: {bit_flip_pos}")

                    new_value = self._bit_flip_value(prev_value, rand_bit)
                    if dim>2:
                        output[i, indices_dim1, indices_dim2, indices_dim3] = new_value
                    else:
                        output[i, indices_dim1] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                dim=len(list(output.size()))
                indices_dim0 = torch.tensor(self.corrupt_batch) # batch
                indices_dim1 = torch.tensor(self.corrupt_dim[0]) # channel
                if(dim>2):
                    indices_dim2 = torch.tensor(self.corrupt_dim[1]) # row
                    indices_dim3 = torch.tensor(self.corrupt_dim[2]) #colum

                if dim>2:
                    prev_value = output[i, indices_dim1, indices_dim2, indices_dim3]
                else:
                    prev_value = output[i, indices_dim1]
                rand_bit = torch.tensor([bit_flip_pos],device=output.device.type)

                logger.info(f"Random Bit: {bit_flip_pos}")

                new_value = self._bit_flip_value(prev_value, rand_bit)
                if dim>2:
                    output[i, indices_dim1, indices_dim2, indices_dim3] = new_value
                else:
                    output[i, indices_dim1] = new_value

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()

        

def random_neuron_single_bit_inj_batched(
    pfi: core.FaultInjection, layer_ranges, batch_random=True
):
    """
    Args:
        pfi: The core.FaultInjection in which the neuron fault injection should be instantiated.
        layer_ranges:
        batch_random (default True): True if each batch should have a random location, if false, then each
                                     batch will use the same randomly generated location.
    """
    pfi.set_conv_max(layer_ranges)

    locations = (
        [random_neuron_location(pfi) for _ in range(pfi.batch_size)]
        if batch_random
        else [random_neuron_location(pfi)] * pfi.batch_size
    )
    # Convert list of tuples [(1, 3), (2, 4)] to list of list [[1, 2], [3, 4]]
    random_layers, random_c, random_h, random_w = map(list, zip(*locations))

    return pfi.declare_neuron_fault_injection(
        batch=range(pfi.batch_size),
        layer_num=random_layers,
        dim1=random_c,
        dim2=random_h,
        dim3=random_w,
        function=pfi.single_bit_flip_signed_across_batch,
    )


def random_neuron_single_bit_inj(pfi: core.FaultInjection, layer_ranges):
    # TODO Support multiple error models via list
    pfi.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)

    return pfi.declare_neuron_fault_injection(
        batch=[batch],
        layer_num=[layer],
        dim1=[C],
        dim2=[H],
        dim3=[W],
        function=pfi.single_bit_flip_signed_across_batch,
    )
