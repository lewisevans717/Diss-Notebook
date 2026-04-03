from compressai.zoo import cheng2020_attn
for q in range(1, 7):
    cheng2020_attn(quality=q, pretrained=True)
    print(f'Level {q} cached.')
print('Done.')
