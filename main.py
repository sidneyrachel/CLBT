import json


def parse_trial_file(filename):
    f = open(filename)
    linex_indexes = f.readlines()

    for i in range(len(linex_indexes)):
        linex_indexes[i] = json.loads(linex_indexes[i].strip())

    print('linex index:', linex_indexes[0]['linex_index'])
    print('len features:', len(linex_indexes[0]['features']))
    print('token:', linex_indexes[0]['features'][0]['token'])
    print('layers:', linex_indexes[0]['features'][0]['layers'])
    print('layers values len:', len(linex_indexes[0]['features'][0]['layers'][0]['values']))
    available_tokens = []
    for feature in linex_indexes[0]['features']:
        available_tokens.append(feature['token'])
    print('tokens:', available_tokens)

    f.close()


def prepare_alignment_file(src, tgt, with_hashtag):
    f = open(src)
    linex_indexes = f.readlines()

    f_write = open(tgt, 'a')

    for i in range(len(linex_indexes)):
        linex_indexes[i] = json.loads(linex_indexes[i].strip())

    for linex_index in linex_indexes:
        tokens = []

        for feature in linex_index['features']:
            token = feature['token']
            if token != '[CLS]' and token != '[SEP]':
                if with_hashtag:
                    tokens.append(token)
                else:
                    if token.startswith('##'):
                        tokens.append(token[2:])
                    else:
                        tokens.append(token)

        f_write.write(' '.join(tokens) + '\n')

    f.close()
    f_write.close()


def generate_alignment_file(src, tgt, filename):
    f_src = open(src, 'r')
    f_tgt = open(tgt, 'r')

    src_lines = f_src.readlines()
    tgt_lines = f_tgt.readlines()

    if len(src_lines) != len(tgt_lines):
        raise Exception(f'Length is not the same. Src: {len(src_lines)}. Tgt: {len(tgt_lines)}.')

    f_res = open(filename, 'w')

    for i in range(len(src_lines)):
        f_res.write(src_lines[i].strip() + ' ||| ' + tgt_lines[i].strip() + '\n')

    f_src.close()
    f_tgt.close()
    f_res.close()


if __name__ == '__main__':
    # parse_trial_file('trial_data/de-en.100.en.bert')
    # parse_trial_file('trial_data/de-en.100.de.bert')
    prepare_alignment_file(
        src='trial_data/de-en.100.de.bert',
        tgt='trial_data/de.raw',
        with_hashtag=False
    )
    prepare_alignment_file(
        src='trial_data/de-en.100.de.bert',
        tgt='trial_data/de.raw.hashtag',
        with_hashtag=True
    )
    prepare_alignment_file(
        src='trial_data/de-en.100.en.bert',
        tgt='trial_data/en.raw',
        with_hashtag=False
    )
    prepare_alignment_file(
        src='trial_data/de-en.100.en.bert',
        tgt='trial_data/en.raw.hashtag',
        with_hashtag=True
    )
    generate_alignment_file(
        src='trial_data/de.raw',
        tgt='trial_data/en.raw',
        filename='trial_data/text.de-en.100.uncased'
    )
    generate_alignment_file(
        src='trial_data/de.raw.hashtag',
        tgt='trial_data/en.raw.hashtag',
        filename='trial_data/text.de-en.100.uncased.hashtag'
    )
