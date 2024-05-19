import datetime
import json
import os
import string


# Chosen to be one token in the OLMo tokenizer. Replace for other tokenizers.
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
WEEKDAYS = ['M', 'Tu', 'W', 'Th', 'F', 'Sa', 'Su']


def load_chat(chat_path):
    with open(chat_path, "r") as fp:
        data = json.load(fp)
    
    data = sorted(data, key=lambda x: x['timestamp_ms'], reverse=False)

    names = sorted(list(set([message['sender_name'] for message in data])))
    assert(len(names) < len(string.ascii_uppercase))
    name_map = {name: string.ascii_uppercase[i] for i, name in enumerate(names)}
    
    data = [
        (name_map[message['sender_name']], int(message['timestamp_ms']), message['content'] if 'content' in message else None) 
        for message in data
    ]

    return data


def load_transcriptions(transcription_dir):
    chats = []
    for f in os.listdir(transcription_dir):
        assert(f.endswith(".tsv"))
        with open(os.path.join(transcription_dir, f), "r") as fp:
            lines = [l.strip() for l in fp.readlines()]

        data = [l.split('\t') for l in lines]
        data = [[w.strip() for w in l] for l in data]
        
        chat = []
        for l in data:
            start, stop, content, speaker = l
            assert(len(content.split()) == 1)
            chat.append((speaker, start, stop, content))
    
        chats.append(chat)

    return chats


def convert_unix(timestamp):
    dt = datetime.datetime.utcfromtimestamp(timestamp/1000)
    return dt


def convert_timestamp(timestamp, old_timestamp=None):
    ret = ""
    dt = convert_unix(timestamp)
    fallthrough = False
    if not old_timestamp or old_timestamp.year != dt.year:
        ret += str(dt.year)
        fallthrough = True
    if not old_timestamp or old_timestamp.month != dt.month or fallthrough:
        #ret += ' ' + MONTHS[dt.month-1]
        ret += MONTHS[dt.month-1]
        fallthrough = True
    if not old_timestamp or old_timestamp.day != dt.day or fallthrough:
        ret += "{:02d}".format(dt.day)
        #ret += ' ' + WEEKDAYS[dt.weekday()]
        ret += WEEKDAYS[dt.weekday()]
        fallthrough = True
    if not old_timestamp or old_timestamp.hour != dt.hour or fallthrough:
        if(not fallthrough):
            ret += '+'
        ret += "{:02d}".format(dt.hour)
        fallthrough = True
    if not old_timestamp or old_timestamp.minute != dt.minute or fallthrough:
        if(not fallthrough):
            ret += ':'
        ret += "{:02d}".format(dt.minute)
        fallthrough = True
    if not old_timestamp or old_timestamp.second != dt.second or fallthrough:
        if(not fallthrough):
            ret += ';'
        ret += "{:02d}".format(dt.second)
        fallthrough = True
    if not old_timestamp or old_timestamp.microsecond != dt.microsecond or fallthrough:
        if(not fallthrough):
            ret += '.'
        ret += str(dt.microsecond//100000)
        fallthrough = True

    return ret


def convert_message(ex, old_timestamp, old_user, delimiter):
    uid, timestamp, content = ex

#    if(ret == old_user):
#        ret = ""

    ret = ''
    ret += convert_timestamp(timestamp, old_timestamp)
    ret += uid
    prefix_len = len(ret)
    ret += (content if content else '')
    ret += delimiter
    return ret, convert_unix(timestamp), prefix_len


def convert_chat(*args, **kwargs):
    return convert_chat_with_datetime(*args, **kwargs)[0]


def convert_chat_with_datetime(messages, delimiter=']--[@'):
    ret = []
    prefix_lens = []
    if len(messages) == 0:
        return []
   
    messages = [m for m in messages if not (m[-1] is None or len(m[-1]) == 0)]

    first_dt = None
    dt = None
    user = None
    for message in messages:
        message = list(message)
        if(delimiter in message[-1]):
            print("WARNING: Delimiter found in message")
            print(message[-1])
            while(delimiter in message[-1]):
                message[-1] = message[-1].replace(delimiter, '')

            #assert(len(message[-1]) != 0)
            if(len(message[-1]) == 0):
                message[-1] = "null"
        
        app_ret, dt, prefix_len = convert_message(message, dt, user, delimiter)
        user = message[0]
        ret.append(app_ret)
        prefix_lens.append(prefix_len)
    
    return [(user, timestamp, pl, ret) for (user, timestamp, _), ret, pl in zip(messages, ret, prefix_lens)], dt 


def convert_transcriptions(chats, mod=10, precision=2, include_stop=False):
    ret = []
    for chat in chats:
        converted_chat = []
        cur_speaker = None
        for word in chat:
            speaker, start, stop, content = word
            start = round(float(start), precision)
            stop = round(float(stop), precision)

            format_string = '{:.' + str(precision) + 'f}'
            start_sec, start_decimal = format_string.format(start).split('.')
            stop_sec, stop_decimal = format_string.format(stop).split('.')

            start_sec = str(int(start_sec) % mod)
            stop_sec = str(int(stop_sec) % mod)

            content = content.lower().strip(string.punctuation)

#            prefix = " "
#            if(cur_speaker is None or cur_speaker != speaker):
#                prefix += string.ascii_uppercase[int(speaker)]
#
#            prefix = f"{prefix}{start_sec}{start_decimal}"
#
#            if(include_stop):
#                prefix = f"{prefix}{stop_sec}{stop_decimal}"

            prefix =  f" {start_sec}{start_decimal}"
            if(include_stop):
                prefix = f"{prefix}{stop_sec}{stop_decimal}"

            prefix = f"{prefix}{string.ascii_uppercase[int(speaker)]}"

            pl = len(prefix)
            converted_chat.append((speaker, start, stop, pl, prefix + content))

            cur_speaker = speaker

        ret.append(converted_chat)

    return ret
