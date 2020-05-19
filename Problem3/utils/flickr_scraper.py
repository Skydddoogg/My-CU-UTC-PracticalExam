# Generated by Glenn Jocher (glenn.jocher@ultralytics.com) for https://github.com/ultralytics

import argparse
import os
import time

import requests
from flickrapi import FlickrAPI

key = '4264e3ade3674464170e517c11b66a0c'  # Flickr API key https://www.flickr.com/services/apps/create/apply
secret = 'dfb3277d3818e0f6'


def download_uri(uri, dir='./'):
    with open(dir + uri.split('/')[-1], 'wb') as f:
        f.write(requests.get(uri, stream=True).content)


def get_urls(search='honeybees on flowers', n=10, download=False):
    print('Fetching images for "{0}"...'.format(search))
    t = time.time()
    flickr = FlickrAPI(key, secret)
    photos = flickr.walk(text=search,  # http://www.flickr.com/services/api/flickr.photos.search.html
                         extras='url_o',
                         per_page=100,
                         sort='relevance')

    if download:
        dir = os.getcwd() + os.sep + 'images' + os.sep + search.replace(' ', '_') + os.sep  # save directory
        if not os.path.exists(dir):
            os.makedirs(dir)

    urls = []

    for i, photo in enumerate(photos):
        if i == n:
            break

        try:
            # construct url https://www.flickr.com/services/api/misc.urls.html
            url = photo.get('url_o')  # original size
            if url is None:
                url = 'https://farm%s.staticflickr.com/%s/%s_%s_b.jpg' % \
                      (photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret'))  # large size

            # download
            if download:
                download_uri(url, dir)

            urls.append(url)
            # print('%g/%g %s' % (i, n, url))
        except:
            pass
            # print('%g/%g error...' % (i, n))

    # import pandas as pd
    # urls = pd.Series(urls)
    # urls.to_csv(search + "_urls.csv")
    # print('Done. (%.1fs)' % (time.time() - t) + ('\nAll images saved to %s' % dir if download else ''))