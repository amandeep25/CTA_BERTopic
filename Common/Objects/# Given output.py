# Given output
topics_output = {
    'Main': {
        -1: [('goose', 0.09386467560567974), ('mr', 0.09327376630987214), ('thank', 0.09096188026984936), ('cs', 0.0288302565331354), ('math', 0.02614796407297852), ('like', 0.023314366704724643), ('transfer', 0.020320718241692067), ('work', 0.017384097798268477), ('people', 0.015594647655511398), ('courses', 0.015576793576165112)],
        0: [('courses', 0.0959412681722576), ('course', 0.08771205056556276), ('math', 0.08535073737342004), ('exam', 0.06221910832041078), ('cs', 0.05347356005752884), ('physics', 0.04580213318529335), ('stat', 0.0442202940005544), ('engineering', 0.0435753815862552), ('research', 0.03616515341537823), ('science', 0.03191995698648622)],
        # More topics...
    }
}

# Convert the output to the desired format
desired_output = {}
for topic, words_weights in topics_output['Main'].items():
    word_weight_list = [(word, weight) for word, weight in words_weights]
    desired_output[topic] = word_weight_list

# Print the desired output
for topic, word_weight_list in desired_output.items():
    print(f"Topic {topic}: {word_weight_list}")
