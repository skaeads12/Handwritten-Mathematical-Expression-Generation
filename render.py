
import math
import random

from PIL import Image, ImageDraw, ImageFont

matrix_types = {
    "p": ["(", ")"],
    "v": ["|", "|"],
    "b": ["[", "]"]
}

def render(font_file, chars, exp_flags=[], base_flags=[], size=(256, 256), pad=20):

    if len(exp_flags) == 0:
        exp_flags = [random.random() > 0.9 for _ in chars]
        for i, exp_flag in enumerate(exp_flags):
            if i == 0:
                continue

            if exp_flag and exp_flags[i-1]:
                exp_flags[i] = False

    if len(base_flags) == 0:
        base_flags = [random.random() > 0.9 for _ in chars]
        for i, base_flag in enumerate(base_flags):
            if i == 0:
                continue

            if base_flag and base_flags[i-1]:
                base_flags[i] = False

    font = ImageFont.truetype(font_file, size=128)
    width, height = font.getsize("".join(chars))

    width += pad * 2
    height += pad * 2

    image = Image.new("L", size=(width, height), color=255)
    draw = ImageDraw.Draw(image)

    x = pad
    y = height - pad

    min_y = height

    for i, (char, exp_flag, base_flag) in enumerate(zip(chars, exp_flags, base_flags)):

        if (exp_flag or base_flag) and i != 0:

            font = ImageFont.truetype(font_file, size=64)

            if exp_flag:
                draw.text((x, y - char_height), char, font=font)
                char_width, char_height = font.getsize(char)
            else:
                char_width, char_height = font.getsize(char)
                draw.text((x, y - char_height), char, font=font)

            if (i + 1) < len(chars) and (exp_flags[i+1] or base_flags[i+1]):
                continue
            else:
                x += char_width
        
        else:

            font = ImageFont.truetype(font_file, size=128)

            char_width, char_height = font.getsize(char)

            draw.text((x, y - char_height), char, font=font)

            min_y = min(min_y, y - char_height)

            x += char_width

    image = image.crop((0, min_y - pad, min(width, x + pad), height))
    width, height = image.size

    if width > height * 8:

        image1 = image.crop((0, 0, width // 2, height))
        image1 = image1.resize(size, resample=Image.BILINEAR)
        image2 = image.crop((width // 2, 0, width, height))
        image2 = image2.resize(size, resample=Image.BILINEAR)
        return [image1, image2]

    else:
        image = image.resize(size, resample=Image.BILINEAR)
        return [image]

def render_matrix(font_file, chars, exp_flags=[], base_flags=[], matrix_type="n", interval=80, size=(256, 256), pad=20):

    if len(exp_flags) == 0:
        exp_flags = [random.random() > 0.9 for _ in chars]
        for i, exp_flag in enumerate(exp_flags):
            if i == 0:
                exp_flags[i] = False
                continue

            if exp_flag and exp_flags[i-1]:
                exp_flags[i] = False

    if len(base_flags) == 0:
        base_flags = [random.random() > 0.9 for _ in chars]
        for i, base_flag in enumerate(base_flags):
            if i == 0:
                base_flags[i] = False
                continue

            if base_flag and base_flags[i-1]:
                base_flags[i] = False

    sig_chars = [char for char, exp_flag, base_flag in zip(chars, exp_flags, base_flags) if not(exp_flag or base_flag)]
    matrix_size = math.ceil(math.sqrt(len(sig_chars)))

    font = ImageFont.truetype(font_file, size=128)
    char_width, char_height = font.getsize("Z")
    
    width = (char_width * matrix_size) + (interval * (matrix_size - 1)) + (pad * 2)
    height = (char_height * matrix_size) + (interval * (matrix_size - 1)) + (pad * 2)

    if matrix_type != "n":

        font = ImageFont.truetype(font_file, size=128 * matrix_size)
        bracket = matrix_types[matrix_type]
        bracket_width, bracket_height = font.getsize(bracket[0])

        width += (bracket_width + pad) * 2
        
        image = Image.new("L", size=(width, height), color=255)
        draw = ImageDraw.Draw(image)

        draw.text((pad, (height - bracket_height) // 2), bracket[0], font=font)
        draw.text((width - pad - bracket_width, (height - bracket_height) // 2), bracket[1], font=font)

        x = pad + bracket_width + pad
        x_gap = interval + char_width

        y = pad + char_height
        y_gap = interval + char_height
    else:
        image = Image.new("L", size=(width, height), color=255)
        draw = ImageDraw.Draw(image)

        bracket_width = 0

        x = pad
        x_gap = interval + char_width

        y = pad + char_height
        y_gap = interval + char_height

    x = [x + (i * x_gap) for i in range(matrix_size)]
    y = [y + (i * y_gap) for i in range(matrix_size)]

    col = 0
    row = 0

    for i, (char, exp_flag, base_flag) in enumerate(zip(chars, exp_flags, base_flags)):
        
        if (exp_flag or base_flag) and i != 0:

            font = ImageFont.truetype(font_file, size=64)

            if exp_flag:
                
                draw.text((x[col] + char_width, y[row] - char_height), char, font=font)

            else:

                _, base_height = font.getsize(char)
                draw.text((x[col] + char_width, y[row] - base_height), char, font=font)

        else:
            
            font = ImageFont.truetype(font_file, size=128)
            char_width, char_height = font.getsize(char)
            draw.text((x[col], y[row] - char_height), char, font=font)

        if (i + 1) < len(chars) and not (exp_flags[i + 1] or base_flags[i + 1]):
            col += 1

        if col == matrix_size:
            col = 0
            row += 1
    
    image = image.resize(size, resample=Image.BILINEAR)

    return [image]

def render_frac(font_file, chars, exp_flags=[], base_flags=[], frac_size=2, size=(256, 256), pad=20):

    if len(exp_flags) == 0:
        exp_flags = [random.random() > 0.9 for _ in chars]
        for i, exp_flag in enumerate(exp_flags):
            if i == 0:
                exp_flags[i] = False
                continue

            if exp_flag and exp_flags[i-1]:
                exp_flags[i] = False

    if len(base_flags) == 0:
        base_flags = [random.random() > 0.9 for _ in chars]
        for i, base_flag in enumerate(base_flags):
            if i == 0:
                base_flags[i] = False
                continue

            if base_flag and base_flags[i-1]:
                base_flags[i] = False

    frac_blocks = [chars]

    while len(frac_blocks) < frac_size:

        temp = []        
        for frac_block in frac_blocks:
            temp.append(frac_block[:len(frac_block) // 2])
            temp.append(frac_block[len(frac_block) // 2:])
        
        frac_blocks = temp

    big_font = ImageFont.truetype(font_file, size=128)
    small_font = ImageFont.truetype(font_file, size=64)

    char_index = 0

    width = 0
    height = 0

    block_width = []
    block_height = []

    bar_width, bar_height = big_font.getsize("-")

    for i, frac_block in enumerate(frac_blocks):

        block_width.append(0)
        block_height.append(0)

        for j, char in enumerate(frac_block):

            if (exp_flags[char_index] or base_flags[char_index]) and j != 0:
                char_width, char_height = small_font.getsize(char)
            else:
                char_width, char_height = big_font.getsize(char)

            block_width[-1] += char_width
            block_height[-1] = max(block_height[-1], char_height)

            char_index += 1

    width = max(block_width) + (pad * 2)
    height = sum(block_height) + (bar_height * (frac_size - 1)) + (pad * (frac_size + 2))
    
    image = Image.new("L", size=(width, height), color=255)
    draw = ImageDraw.Draw(image)

    char_index = 0
    bar_size = [0 for _ in range(frac_size)]

    while frac_size > 1:

        frac_size = frac_size // 2

        for i in range(0, len(bar_size), frac_size):
            bar_size[i] += 1

    bar_size = bar_size[1:]

    for i, frac_block in enumerate(frac_blocks):

        x = (width - block_width[i]) // 2
        y = sum(block_height[:i+1]) + (i * bar_height) + ((i + 1) * pad)

        for j, char in enumerate(frac_block):

            if (exp_flags[char_index] or base_flags[char_index]) and j != 0:
                
                font = ImageFont.truetype(font_file, size=64)

                if exp_flags[char_index]:
                    draw.text((x, y - char_height), char, font=font)
                    char_width, char_height = font.getsize(char)
                else:
                    char_width, char_height = font.getsize(char)
                    draw.text((x, y - char_height), char, font=font)

                if (char_index + 1) < len(chars) and not(exp_flags[char_index + 1] or base_flags[char_index + 1]):
                    x += char_width

            else:
                
                font = ImageFont.truetype(font_file, size=128)

                char_width, char_height = font.getsize(char)

                draw.text((x, y - char_height), char, font=font)
                
                x += char_width

            char_index += 1

        if frac_block != frac_blocks[-1]:

            font = ImageFont.truetype(font_file, size=256)
            bar_width, _ = font.getsize("-" * bar_size[i])

            draw.text(((width - bar_width) // 2, (y - bar_height)), "-" * bar_size[i], font=font)

    width, height = image.size

    if height > width * 5:

        image1 = image.crop((0, 0, width, height // 2))
        image1 = image1.resize(size, resample=Image.BILINEAR)
        image2 = image.crop((0, height // 2, width, height))
        image2 = image2.resize(size, resample=Image.BILINEAR)

        return [image1, image2]

    else:
        image = image.resize(size, resample=Image.BILINEAR)
        return [image]