import numpy as np

available_weights = ['ffhq', 'car', 'cat', 'church', 'horse']
weights_stylegan2_dir = 'weights/'

mapping_weights = [ 'Dense0_weight', 'Dense0_bias',
                    'Dense1_weight', 'Dense1_bias',
                    'Dense2_weight', 'Dense2_bias',
                    'Dense3_weight', 'Dense3_bias',
                    'Dense4_weight', 'Dense4_bias',
                    'Dense5_weight', 'Dense5_bias',
                    'Dense6_weight', 'Dense6_bias',
                    'Dense7_weight', 'Dense7_bias']

def get_synthesis_name_weights(resolution):
    synthesis_weights = ['4x4_Const_const',
                         '4x4_Conv_noise_strength',
                         '4x4_Conv_bias',
                         '4x4_Conv_mod_bias',
                         '4x4_Conv_mod_weight',
                         '4x4_Conv_weight',
                         '4x4_ToRGB_bias',
                         '4x4_ToRGB_mod_bias',
                         '4x4_ToRGB_mod_weight',
                         '4x4_ToRGB_weight']

    for res in range(3,int(np.log2(resolution)) + 1):
        name = '{}x{}_'.format(2**res, 2**res)
        for up in ['Conv0_up_', 'Conv1_', 'ToRGB_']:
            for var in ['noise_strength', 'bias', 'mod_bias', 'mod_weight', 'weight']:
                if up == 'ToRGB_' and var == 'noise_strength':
                    continue
                synthesis_weights.append(name+up+var)
                
    return synthesis_weights

synthesis_weights_1024 = get_synthesis_name_weights(1024)
synthesis_weights_512 = get_synthesis_name_weights(512)
synthesis_weights_256 = get_synthesis_name_weights(256)


discriminator_weights_1024 = ['disc_4x4_Conv_bias',
                            'disc_1024x1024_FromRGB_bias',
                            'disc_1024x1024_FromRGB_weight',
                            'disc_1024x1024_Conv0_bias',
                            'disc_1024x1024_Conv1_down_bias',
                            'disc_1024x1024_Conv0_weight',
                            'disc_1024x1024_Conv1_down_weight',
                            'disc_1024x1024_Skip_weight',
                            'disc_512x512_Conv0_bias',
                            'disc_512x512_Conv1_down_bias',
                            'disc_512x512_Conv0_weight',
                            'disc_512x512_Conv1_down_weight',
                            'disc_512x512_Skip_weight',
                            'disc_256x256_Conv0_bias',
                            'disc_256x256_Conv1_down_bias',
                            'disc_256x256_Conv0_weight',
                            'disc_256x256_Conv1_down_weight',
                            'disc_256x256_Skip_weight',
                            'disc_128x128_Conv0_bias',
                            'disc_128x128_Conv1_down_bias',
                            'disc_128x128_Conv0_weight',
                            'disc_128x128_Conv1_down_weight',
                            'disc_128x128_Skip_weight',
                            'disc_64x64_Conv0_bias',
                            'disc_64x64_Conv1_down_bias',
                            'disc_64x64_Conv0_weight',
                            'disc_64x64_Conv1_down_weight',
                            'disc_64x64_Skip_weight',
                            'disc_32x32_Conv0_bias',
                            'disc_32x32_Conv1_down_bias',
                            'disc_32x32_Conv0_weight',
                            'disc_32x32_Conv1_down_weight',
                            'disc_32x32_Skip_weight',
                            'disc_16x16_Conv0_bias',
                            'disc_16x16_Conv1_down_bias',
                            'disc_16x16_Conv0_weight',
                            'disc_16x16_Conv1_down_weight',
                            'disc_16x16_Skip_weight',
                            'disc_8x8_Conv0_bias',
                            'disc_8x8_Conv1_down_bias',
                            'disc_8x8_Conv0_weight',
                            'disc_8x8_Conv1_down_weight',
                            'disc_8x8_Skip_weight',
                            'disc_4x4_Conv_weight',
                            'disc_4x4_Dense0_weight',
                            'disc_4x4_Dense0_bias',
                            'disc_Output_weight',
                            'disc_Output_bias']

discriminator_weights_512 = ['disc_4x4_Conv_bias',
                            'disc_512x512_FromRGB_bias',
                            'disc_512x512_FromRGB_weight',
                            'disc_512x512_Conv0_bias',
                            'disc_512x512_Conv1_down_bias',
                            'disc_512x512_Conv0_weight',
                            'disc_512x512_Conv1_down_weight',
                            'disc_512x512_Skip_weight',
                            'disc_256x256_Conv0_bias',
                            'disc_256x256_Conv1_down_bias',
                            'disc_256x256_Conv0_weight',
                            'disc_256x256_Conv1_down_weight',
                            'disc_256x256_Skip_weight',
                            'disc_128x128_Conv0_bias',
                            'disc_128x128_Conv1_down_bias',
                            'disc_128x128_Conv0_weight',
                            'disc_128x128_Conv1_down_weight',
                            'disc_128x128_Skip_weight',
                            'disc_64x64_Conv0_bias',
                            'disc_64x64_Conv1_down_bias',
                            'disc_64x64_Conv0_weight',
                            'disc_64x64_Conv1_down_weight',
                            'disc_64x64_Skip_weight',
                            'disc_32x32_Conv0_bias',
                            'disc_32x32_Conv1_down_bias',
                            'disc_32x32_Conv0_weight',
                            'disc_32x32_Conv1_down_weight',
                            'disc_32x32_Skip_weight',
                            'disc_16x16_Conv0_bias',
                            'disc_16x16_Conv1_down_bias',
                            'disc_16x16_Conv0_weight',
                            'disc_16x16_Conv1_down_weight',
                            'disc_16x16_Skip_weight',
                            'disc_8x8_Conv0_bias',
                            'disc_8x8_Conv1_down_bias',
                            'disc_8x8_Conv0_weight',
                            'disc_8x8_Conv1_down_weight',
                            'disc_8x8_Skip_weight',
                            'disc_4x4_Conv_weight',
                            'disc_4x4_Dense0_weight',
                            'disc_4x4_Dense0_bias',
                            'disc_Output_weight',
                            'disc_Output_bias']

discriminator_weights_256 =  ['disc_4x4_Conv_bias',
                            'disc_256x256_FromRGB_bias',
                            'disc_256x256_FromRGB_weight',
                            'disc_256x256_Conv0_bias',
                            'disc_256x256_Conv1_down_bias',
                            'disc_256x256_Conv0_weight',
                            'disc_256x256_Conv1_down_weight',
                            'disc_256x256_Skip_weight',
                            'disc_128x128_Conv0_bias',
                            'disc_128x128_Conv1_down_bias',
                            'disc_128x128_Conv0_weight',
                            'disc_128x128_Conv1_down_weight',
                            'disc_128x128_Skip_weight',
                            'disc_64x64_Conv0_bias',
                            'disc_64x64_Conv1_down_bias',
                            'disc_64x64_Conv0_weight',
                            'disc_64x64_Conv1_down_weight',
                            'disc_64x64_Skip_weight',
                            'disc_32x32_Conv0_bias',
                            'disc_32x32_Conv1_down_bias',
                            'disc_32x32_Conv0_weight',
                            'disc_32x32_Conv1_down_weight',
                            'disc_32x32_Skip_weight',
                            'disc_16x16_Conv0_bias',
                            'disc_16x16_Conv1_down_bias',
                            'disc_16x16_Conv0_weight',
                            'disc_16x16_Conv1_down_weight',
                            'disc_16x16_Skip_weight',
                            'disc_8x8_Conv0_bias',
                            'disc_8x8_Conv1_down_bias',
                            'disc_8x8_Conv0_weight',
                            'disc_8x8_Conv1_down_weight',
                            'disc_8x8_Skip_weight',
                            'disc_4x4_Conv_weight',
                            'disc_4x4_Dense0_weight',
                            'disc_4x4_Dense0_bias',
                            'disc_Output_weight',
                            'disc_Output_bias']

synthesis_weights = {
    'ffhq' : synthesis_weights_1024,
    'car' : synthesis_weights_512,
    'cat' : synthesis_weights_256,
    'horse' : synthesis_weights_256,
    'church' : synthesis_weights_256
    }

discriminator_weights = {
    'ffhq' : discriminator_weights_1024,
    'car' : discriminator_weights_512,
    'cat' : discriminator_weights_256,
    'horse' : discriminator_weights_256,
    'church' : discriminator_weights_256
    }
