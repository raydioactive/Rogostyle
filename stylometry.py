import re
import string

def occurences(sentence, words):
    end = r'\b'
    regex = end + '(' + '|'.join(words) + ')' + end
    return len(re.findall(regex, sentence, re.IGNORECASE))

def multiOccurence(sentence, listArray):
    count = 0
    for word in sentence.lower().split():
        if word in listArray or word.endswith('que'):
            count += 1
    return count

def startWith(sentence, list_):
    return any(sentence.startswith(word) for word in list_)

def endWith(sentence, list_):
    count = 0
    for word in list_:
        if sentence.endswith(word):
            count += 1
    return count

def interrogative(sentence):
    return sentence.count('?')

def countWords(sentence):
    return len(sentence.split())

def sigFig(num):
    return round(num, 5) if num else 0

def startf(sentence, consonant):
    count = 0
    words = sentence.lower().split()
    for i in range(len(words)):
        if words[i] == 'atque' and startWith(words[(i + 1) % len(words)], consonant):
            count += 1
    return count

def charCount(sentence):
    return len(re.sub(r'[^a-zA-Z0-9]', '', sentence))

def relative(sentence, list_):
    count = 0
    sent_count = 0
    relative_array = []
    if '!' in sentence or '.' in sentence:
        sent_count += 1
    if ('!' in sentence and occurences(sentence, list_) >= 1) or ('.' in sentence and occurences(sentence, list_) >= 1):
        all_array = re.findall(r'\b(' + '|'.join(list_) + r')\b', sentence, re.IGNORECASE)
        merge_array = list_ + [',', ':', '.', '!', ';']
        for item in all_array:
            found_indexes = [sentence.find(x, item.end()) for x in merge_array if sentence.find(x, item.end()) != -1 and sentence.find(x, item.end()) != item.start()]
            found_indexes = min(found_indexes) if found_indexes else len(sentence)
            relative_array.append(sentence[item.start():found_indexes])
        count += 1
    final_string = ''.join(relative_array)
    string_count = len(re.sub(r'[^a-zA-Z0-9]', '', final_string))
    return {
        'relative': count / sent_count if sent_count > 0 else 0,
        'mean': string_count / len(relative_array) if relative_array else 0
    }

def removePunct(sentence):
    return re.sub(r'[^a-zA-Z0-9\n\r\s]', ' ', sentence)

def endf(sentence, vocatives):
    count = 0
    words = removePunct(sentence).split()
    for i in range(len(words)):
        first = words[i].lower()
        second = words[(i + 1) % len(words)]
        if first == 'o' and endWith(second, vocatives):
            count += 1
    return count

def cumClause(sentence, cum_clauses):
    count = 0
    words = removePunct(sentence).lower().split()
    for i in range(len(words)):
        if words[i] == 'cum' and not endWith(words[(i + 1) % len(words)], cum_clauses):
            count += 1
    return count

def endGerund(sentence, gerund):
    count = 0
    words = removePunct(sentence).lower().split()
    for word in words:
        if endWith(word, gerund) and word != 'nondum':
            count += 1
    return count

def superlatives(sentence):
    count = 0
    for word in sentence.lower().split():
        if 'issim' in word:
            count += 1
    return count

def meanSentence(sentences):
    return sum(len(re.sub(r'[^a-zA-Z0-9]', '', sentence).strip()) for sentence in sentences) / len(sentences)

import re

def calculate_stylometry(sentence):
    book = sentence
    ngram_removed = re.sub(r'[""";:&,\.\/?\\-]', '', book).strip()
    words = countWords(book)
    characters = charCount(book)

    personal_pronoun = ['ego', 'mei', 'mihi', 'me', 'tu', 'tui', 'tibi', 'te', 'nos', 'nostri', 'nobis', 'vos', 'vestri', 'vobis', 'uos', 'uestri', 'uobis', 'mi', 'nostrum', 'vestrum', 'vostrum', 'vostri', 'uestrum', 'uostrum', 'uostri', 'egoque', 'meique', 'mihique', 'meque', 'tuque', 'tuique', 'tibique', 'teque', 'nosque', 'nostrique', 'nobisque', 'vosque', 'vestrique', 'vobisque', 'uosque', 'uestrique', 'uobisque', 'mique', 'nostrumque', 'vestrumque', 'vostrumque', 'vostrique', 'uestrumque', 'uostrumque', 'uostrique']
    demonstrative_pronoun = ['hic', 'hunc', 'huius', 'huic', 'hoc', 'haec', 'hanc', 'hac', 'hi', 'hos', 'horum', 'his', 'hae', 'has', 'harum', 'ho', 'ha', 'ille', 'illum', 'illius', 'illi', 'illo', 'illa', 'illam', 'illud', 'illos', 'illorum', 'illis', 'illas', 'illarum', 'illae', 'is', 'eum', 'eius', 'ei', 'eo', 'ea', 'eam', 'id', 'ii', 'eos', 'eorum', 'eis', 'iis', 'eae', 'eas', 'earum', 'illic', 'illaec', 'illuc', 'illic', 'illoc', 'illunc', 'illanc', 'illac', 'hicque', 'huncque', 'huiusque', 'huicque', 'hocque', 'haecque', 'hancque', 'hacque', 'hique', 'hosque', 'horumque', 'hisque', 'haeque', 'hasque', 'harumque', 'hoque', 'haque', 'illeque', 'illumque', 'illiusque', 'illique', 'illoque', 'illaque', 'illamque', 'illudque', 'illosque', 'illorumque', 'illisque', 'illasque', 'illarumque', 'illaeque', 'isque', 'eumque', 'eiusque', 'eique', 'eoque', 'eaque', 'eamque', 'idque', 'iique', 'eosque', 'eorumque', 'eisque', 'iisque', 'eaeque', 'easque', 'earumque', 'illicque', 'illaecque', 'illucque', 'illicque', 'illocque', 'illuncque', 'illancque', 'illacque']
    quidam = ['quidam', 'quendam', 'cuiusdam', 'cuidam', 'quodam', 'quaedam', 'quandam', 'quodam', 'quoddam', 'quosdam', 'quorundam', 'quibusdam', 'quasdam', 'quarundam', 'quiddam', 'quoddam', 'quadam', 'quidamque', 'quendamque', 'cuiusdamque', 'cuidamque', 'quodamque', 'quaedamque', 'quandamque', 'quodamque', 'quoddamque', 'quosdamque', 'quorundamque', 'quibusdamque', 'quasdamque', 'quarundamque', 'quiddamque', 'quoddamque', 'quadamque']
    reflexive_pronoun = ['se', 'sibi', 'sese', 'sui', 'seque', 'sibique', 'seseque', 'suique']
    iste = ['iste', 'istum', 'istius', 'isti', 'isto', 'ista', 'istam', 'istud', 'istos', 'istorum', 'istis', 'istas', 'istarum', 'iste', 'istum', 'istius', 'isti', 'isto', 'ista', 'istam', 'istud', 'istos', 'istorum', 'istis', 'istas', 'istarum', 'isteque', 'istumque', 'istiusque', 'istique', 'istoque', 'istaque', 'istamque', 'istudque', 'istosque', 'istorumque', 'istisque', 'istasque', 'istarumque']
    alius = ['alius', 'alium', 'alii', 'alio', 'alia', 'aliam', 'aliud', 'alios', 'aliorum', 'aliis', 'aliae', 'alias', 'aliarum', 'aliusque', 'aliumque', 'aliique', 'alioque', 'aliaque', 'aliamque', 'aliudque', 'aliosque', 'aliorumque', 'aliisque', 'aliaeque', 'aliasque', 'aliarumque']
    ipse = ['ipse', 'ipsum', 'ipsius', 'ipsi', 'ipso', 'ipsa', 'ipsam', 'ipsos', 'ipsorum', 'ipsas', 'ipsarum', 'ipseque', 'ipsumque', 'ipsiusque', 'ipsique', 'ipsoque', 'ipsaque', 'ipsamque', 'ipsosque', 'ipsorumque', 'ipsasque', 'ipsarumque']
    idem = ['idem', 'eundem', 'eiusdem', 'eidem', 'eodem', 'eadem', 'eandem', 'iidem', 'eosdem', 'eorundem', 'eisdem', 'iisdem', 'eaedem', 'eedem', 'easdem', 'earumdem', 'isdem', 'isdemque', 'idemque', 'eundemque', 'eiusdemque', 'eidemque', 'eodemque', 'eademque', 'eandemque', 'iidemque', 'eosdemque', 'eorundemque', 'eisdemque', 'iisdemque', 'eaedemque', 'easdemque', 'earundemque']
    priu = ['priusquam', 'prius quam']
    anteq = ['antequam', 'ante quam']
    quom = ['quominus', 'quo minus']
    dum = ['dum', 'dumque']
    quin = ['quin']
    ut = ['ut', 'utei', 'utque']
    conditional_clauses = ['si', 'nisi', 'quodsi', 'sin', 'dummodo']
    prepositions = ['ab', 'abs', 'e', 'ex', 'apud', 'de', 'cis', 'erga', 'inter', 'ob', 'penes', 'per', 'praeter', 'propter', 'trans', 'absque', 'pro', 'tenus', 'sub', 'aque', 'abque', 'eque', 'exque', 'apudque', 'deque', 'cisque', 'ergaque', 'interque', 'obque', 'penesque', 'perque', 'praeterque', 'propterque', 'transque', 'proque', 'tenusque', 'subque']
    relatives = ['qui', 'cuius', 'cui', 'quem', 'quo', 'quae', 'quam', 'qua', 'quod', 'quorum', 'quibus', 'quos', 'quarum', 'quas']
    gerund = ['ndum', 'ndus', 'ndorum', 'ndarum', 'ndumque', 'ndusque', 'ndorumque', 'ndarumque']
    o = ['o']
    vocatives = ['e', 'i', 'a', 'u', 'ae', 'es', 'um', 'us']
    cum_clauses = ['a', 'e', 'i', 'o', 'u', 'is', 'ibus', 'ebus', 'obus', 'ubus']
    consonant = ['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
    conjunctions = ['et', 'atque', 'ac', 'aut', 'vel', 'uel', 'at', 'autem', 'sed', 'postquam', 'ast', 'donec', 'dum', 'dummodo', 'enim', 'etiam', 'etiamtum', 'etiamtunc', 'etenim', 'veruntamen', 'ueruntamen', 'uerumtamen', 'verumtamen', 'utrumnam', 'set', 'quocirca', 'quia', 'quamquam', 'quanquam', 'nam', 'namque', 'nanque', 'nempe', 'dumque', 'etiamque', 'quiaque']
    conjunctions_in = ['que']

    gramr = {}

    gramr['personal_pronoun'] = sigFig(occurences(book, personal_pronoun) / characters)
    gramr['demonstrative_pronoun'] = sigFig(occurences(book, demonstrative_pronoun) / characters)
    gramr['quidam'] = sigFig(occurences(book, quidam) / characters)
    gramr['reflexive_pronoun'] = sigFig(occurences(book, reflexive_pronoun) / characters)
    gramr['iste'] = sigFig(occurences(book, iste) / characters)
    gramr['alius'] = sigFig(occurences(book, alius) / characters)
    gramr['ipse'] = sigFig(occurences(book, ipse) / characters)
    gramr['idem'] = sigFig(occurences(book, idem) / characters)
    gramr['priu'] = sigFig(occurences(book, priu) / characters)
    gramr['anteq'] = sigFig(occurences(book, anteq) / characters)
    gramr['quom'] = sigFig(occurences(book, quom) / characters)
    gramr['dum'] = sigFig(occurences(book, dum) / characters)
    gramr['quin'] = sigFig(occurences(book, quin) / characters)
    gramr['ut'] = sigFig(occurences(book, ut) / characters)
    gramr['conditional_clauses'] = sigFig(occurences(book, conditional_clauses) / characters)
    gramr['prepositions'] = sigFig(occurences(book, prepositions) / characters)
    gramr['interrogative'] = sigFig(interrogative(book) / characters)
    gramr['superlatives'] = sigFig(superlatives(book) / characters)
    gramr['consonant'] = sigFig(startf(book, consonant) / characters)

    relative_result = relative(book.split('.'), relatives)
    gramr['relative'] = sigFig(relative_result['relative'])
    gramr['relative_mean'] = sigFig(relative_result['mean'])
    
    gramr['gerund'] = sigFig(endGerund(book, gerund) / characters)
    gramr['cum_clauses'] = sigFig(cumClause(book, cum_clauses) / characters)
    gramr['conjunctions'] = sigFig(multiOccurence(ngram_removed, conjunctions + conjunctions_in) / characters)
    gramr['vocatives'] = sigFig(endf(ngram_removed, vocatives) / characters)
    gramr['meanSentence'] = sigFig(meanSentence(book.split('.')))
    gramr['characters'] = characters
    gramr['words'] = words

    return gramr

