#Script to visualize character by chacracter sentiment

import numpy as np


class Color:
    def __init__(self, _r, _g, _b):
        self.r = _r;
        self.g = _g;
        self.b = _b;


def interpolate(start, end, steps, count):
    s = start
    e = end
    final = s + (((e - s) / steps) * count)
    return np.floor(final)


def generate_sentiment_html(sentiment_array, string, file_name):
    ###
    #  character_sentiment_array: sentiment value for each character in string
    #

    orig_max = max(sentiment_array)
    orig_min = min(sentiment_array)

    f = open(file_name, 'w')
    for x in range(1, len(string)):
        o_val = sentiment_array[x - 1]

        if o_val > 0:
            val = rescale_val(o_val, orig_max, orig_min, 100, 50)
        else:
            val = rescale_val(o_val, orig_max, orig_min, 50, 0)

        html_style, val = generate_letter_html(val)
        f.write('<span o_val="%f" value="%f" %s>%s</span>' % (o_val, val, html_style, string[x]))


def rescale_val(o_val, orig_max, orig_min, new_max, new_min):
    return (((new_max - new_min)*(o_val - orig_min)) / (orig_max - orig_min)) + new_min


def generate_letter_html(val):

    red = Color(232, 9, 26)
    white = Color(255, 255, 255)
    green = Color(6, 170, 60)

    start = red
    end = white

    if val > 50:
        start = white
        end = green
        val = val % 51

    start_colors = start
    end_colors = end
    r = interpolate(start_colors.r, end_colors.r, 26, val)
    g = interpolate(start_colors.g, end_colors.g, 26, val)
    b = interpolate(start_colors.b, end_colors.b, 26, val)
    html_style = 'style = "background-color: rgb(%d, %d, %d);"' % (r, g, b)
    return html_style, val


generate_sentiment_html(feat, s, 'C:/Users/matthew li yuen fong/Desktop/sentiment-analysis-master/data/raw/openaiscore.html')
