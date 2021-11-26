import os
menu_options = {
    1: 'Add all movie in file CSV',
    2: 'Delete movie by id',
    3: 'Add one movie by id',
    4: 'Update movie by id',
    5: 'Exit',
}

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

def option1():
    print('----------Add all movie in file CSV----------')
    os.system('python add_movie.py')

def option2():
    print('---------Delete movie by id----------')
    os.system('python delete_movie.py')

def option3():
    print('----------Add one movie by id----------')
    os.system('python add_one_movie.py')

def option4():
    print('----------Update movie by id----------')
    os.system('python update_movie.py') 


if __name__ == '__main__':
    while(True):
        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        #Check what choice was entered and act accordingly
        if option == 1:
           option1()
        elif option == 2:
            option2()
        elif option == 3:
            option3()
        elif option == 4:
            option4()
        elif option == 5:
            print('Thanks message before exiting')
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 4.')
    
