import codecs

def write_predictions(predict_examples, result, output_predict_file, id2label):
  with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
    for predict_example, prediction in zip(predict_examples, result):
      tokens = predict_example.tokens
      poss = predict_example.poss
      chunks = predict_example.chunks
      labels = predict_example.labels
      tokenized_tokens = predict_example.tokenized_tokens
      tokenized_poss = predict_example.tokenized_poss
      tokenized_chunks = predict_example.tokenized_chunks
      tokenized_labels = predict_example.tokenized_labels
      text = predict_example.text
      length = len(tokenized_tokens)
      seq = 0
      for token, pos, label, p_id in zip(tokenized_tokens, tokenized_poss, tokenized_labels,
                                         prediction[1:length + 1]):
        p_label = 'O'
        if p_id != 0: p_label = id2label[p_id]
        if p_label == 'X': p_label = 'O'
        if label == 'X': continue
        org_token = tokens[seq]
        org_pos = poss[seq]
        org_chunk = chunks[seq]
        org_label = labels[seq]
        output_line = ' '.join([org_token, org_pos, org_chunk, org_label, p_label])
        writer.write(output_line + '\n')
        seq += 1
      writer.write('\n')


def render_predictions(predict_examples, result, id2label):
  for predict_example, prediction in zip(predict_examples, result):
    tokens = predict_example.tokens
    poss = predict_example.poss
    chunks = predict_example.chunks
    labels = predict_example.labels
    tokenized_tokens = predict_example.tokenized_tokens
    tokenized_poss = predict_example.tokenized_poss
    tokenized_chunks = predict_example.tokenized_chunks
    tokenized_labels = predict_example.tokenized_labels
    text = predict_example.text
    length = len(tokenized_tokens)
    seq = 0
    for token, pos, label, p_id in zip(tokenized_tokens, tokenized_poss, tokenized_labels,
                                       prediction[1:length + 1]):
      p_label = 'O'
      if p_id != 0: p_label = id2label[p_id]
      if p_label == 'X': p_label = 'O'
      if label == 'X': continue
      org_token = tokens[seq]
      org_pos = poss[seq]
      org_chunk = chunks[seq]
      org_label = labels[seq]
      output_line = ' '.join([org_token, org_pos, org_chunk, org_label, p_label])
      if p_label == 'O':
        if org_token == '.':
          print('.')
        else:
          print(org_token, end=' ')
      else:
        print(f'{org_token}<{p_label}>', end=' ')
      seq += 1


#         print('\n')

def render_styles():
  return """
<style> 
.x1 {
	padding: .2em .3em;
    padding-top: 0.2em;
    padding-right: 0.3em;
    padding-bottom: 0.2em;
    padding-left: 0.3em;
    margin: 0 .25em;
    margin-top: 0px;
    margin-right: 0.25em;
    margin-bottom: 0px;
    margin-left: 0.25em;
    line-height: 1;
    display: inline-block;
    border-radius: .25em;
    border-top-left-radius: 0.25em;
    border-top-right-radius: 0.25em;
    border-bottom-right-radius: 0.25em;
    border-bottom-left-radius: 0.25em;
}

.x2 {
    box-sizing: border-box;
    content: attr(data-entity);
    font-size: .55em;
    line-height: 1;
    padding: .35em .35em;
    padding-top: 0.35em;
    padding-right: 0.35em;
    padding-bottom: 0.35em;
    padding-left: 0.35em;
    border-radius: .35em;
    text-transform: uppercase;
    display: inline-block;
    vertical-align: middle;
    margin: 0 0 .15rem .5rem;
    margin-top: 0px;
    margin-right: 0px;
    margin-bottom: 0.15rem;
    margin-left: 0.5rem;
    background: #fff;
    background-image: initial;
    background-position-x: initial;
    background-position-y: initial;
    background-size: initial;
    background-repeat-x: initial;
    background-repeat-y: initial;
    background-attachment: initial;
    background-origin: initial;
    background-clip: initial;
    background-color: rgb(255, 255, 255);
    font-weight: 700;
}

</style>
"""


def render_entity(token, label):
  return f'<mark class="x1" style="background-color: rgb(166, 226, 45);">{token}<sub class="x2">{label[2:]}</sub></mark>'


def render_predictions_html(predict_examples, result, id2label):
  str = render_styles()
  prev_label = None
  for predict_example, prediction in zip(predict_examples, result):
    tokens = predict_example.tokens
    poss = predict_example.poss
    chunks = predict_example.chunks
    labels = predict_example.labels
    tokenized_tokens = predict_example.tokenized_tokens
    tokenized_poss = predict_example.tokenized_poss
    tokenized_chunks = predict_example.tokenized_chunks
    tokenized_labels = predict_example.tokenized_labels
    text = predict_example.text
    length = len(tokenized_tokens)
    seq = 0
    for token, pos, label, p_id in zip(tokenized_tokens, tokenized_poss, tokenized_labels,
                                       prediction[1:length + 1]):
      p_label = 'O'
      if p_id != 0: p_label = id2label[p_id]
      if p_label == 'X': p_label = 'O'
      if label == 'X': continue
      org_token = tokens[seq]
      if p_label == 'O':
        if org_token == '.':
          str += '. '
        else:
          str += org_token + ' '
      else:
        str += render_entity(org_token, p_label)
      seq += 1
  return str


def render_predictions_html_v2(predict_examples, result, id2label):
  str = render_styles()
  prev_label = None
  prev_token = None
  entities = []

  def add_e(t, l):
    if prev_label == l:
      (oldtoken, oldlabel) = entities.pop()
      entities.append((oldtoken + ' ' + t, l))
    else:
      entities.append((t, l))

  for predict_example, prediction in zip(predict_examples, result):
    tokens = predict_example.tokens
    labels = predict_example.labels
    tokenized_tokens = predict_example.tokenized_tokens
    tokenized_labels = predict_example.tokenized_labels
    text = predict_example.text
    length = len(tokenized_tokens)
    seq = 0
    for token, label, p_id in zip(tokenized_tokens, tokenized_labels, prediction[1:length + 1]):
      p_label = 'O'
      if p_id != 0: p_label = id2label[p_id]
      if p_label == 'X': p_label = 'O'
      if label == 'X': continue
      org_token = tokens[seq]

      add_e(org_token + ' ', p_label)

      prev_label = p_label
      prev_token = org_token
      seq += 1

  for e in entities:
    if e[1] == 'O':
      str += e[0]
    else:
      str += render_entity(e[0], e[1])
  return str
