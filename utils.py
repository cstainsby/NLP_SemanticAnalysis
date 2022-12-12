
def find_frequency_of_each_element_in_list(input_list):
    """finds each unique value and the frequency of that value

        Args:
            input_list(list): list of data
        Returns:
            item_label_list(list): all unique values found
            parallel_frequency_list(list): the frequencies of those unique values
    """
    item_label_list = []            # name of the elements that were encountered
    parallel_frequency_list = []    # frequencies of each of the elements

    for item in input_list:
        if item_label_list.count(item) == 0:
            item_label_list.append(item)
            parallel_frequency_list.append(1)
        else:
            existing_item_index = item_label_list.index(item)
            parallel_frequency_list[existing_item_index] += 1

    return item_label_list, parallel_frequency_list