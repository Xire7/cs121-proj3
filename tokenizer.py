
"""
contains document ids, for each id:
        total frequency, position_front_frequency, position_middle_frequency, position_end_frequency
        purpose: if query is only phrase that matches document_id, grab the total frequency
            else get the position of the token in the query and cmp against the other three nums
"""
class WordFrequencyAnalyzer:
    @staticmethod
    def tokenize(text_file_path):
        """
        Reads a text file and returns a list of tokens.
        A token is a sequence of alphanumeric characters, case-insensitive.
        """
        tokens = []
        with open(text_file_path, 'r', encoding='utf-8') as file:
            
            for line in file:
                chars = list(line)

                for index,item in enumerate(chars):
                    if item.isalnum()==False:
                            chars[index]=' '
                        
                modified="".join(chars)
                print(modified)
               
               
                # Turn any non-alphanumeric characters to SPACES " ".
                words = modified.split()
                for word in words:
                    # Remove non-alphanumeric characters and convert to lowercase
                    token = ''.join(char.lower() for char in word if char.isalnum())
                    if token:
                        tokens.append(token)
        return tokens

    @staticmethod
    def compute_word_frequencies(token_list):
        """
        Counts the number of occurrences of each token in the list.
        Returns a dictionary with tokens as keys and their frequencies as values.
        """
        frequencies = {}
        for token in token_list:
            frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    @staticmethod
    def print_word_frequencies(frequencies):
        """
        Prints word frequency counts in decreasing order.
        Resolves ties alphabetically and in ascending order.
        """
        sorted_frequencies = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
        for token, count in sorted_frequencies:
            print(f"{token}\t{count}")


# Main execution when the script is run
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python PartA.py [filename]")
        sys.exit(1)

    filename = sys.argv[1]
    tokens = WordFrequencyAnalyzer.tokenize(filename)
    frequencies = WordFrequencyAnalyzer.compute_word_frequencies(tokens)
    WordFrequencyAnalyzer.print_word_frequencies(frequencies)


# PartA.py

# Tokenization:
# The runtime complexity of tokenization is O(N), where N is the total number of characters in the input file.
# This is because each character is processed once, and the alphanumeric characters are extracted.

# Word Frequency Computation:
# The runtime complexity of computing word frequencies is O(M), where M is the number of tokens in the list.
# This is because each token is processed once, and a dictionary is used for efficient frequency counting.

# Printing Word Frequencies:
# The runtime complexity of printing word frequencies is O(K log K), where K is the number of unique tokens.
# This is due to the sorting operation based on frequency and token.
