import anonymizer
    
ANONYMIZATION_STRENGTH = [ 10, 15, 25,  35 ]

def main():
    t_input = input('Wirte the method for the anonymization [blur, mask, pixelate, noise]:')
    METHOD = str(t_input)

    for i in ANONYMIZATION_STRENGTH:
        # print(f"anonymized {i} with {METHOD}")
        anonymizer.anonymise(METHOD, i)

if __name__ == '__main__':
    main()
