#Simple Slim: Losing Weight Program_Shuang (Shirley) Wang
accounts = []
records = []

import random
def BMI_Refer():
    print('Friendly tips:')
    print('BMI Categories Reference:')
    print('Underweight <18.5')
    print('Normal weight 18.5–24.9')
    print('Overweight 25–29.9')
    print('Obesity 30 or greater')
    
def sign_up():
    print('Welcome to Simple Slim')
    print('Basic Information:')
    acct = int(random.uniform(0, 10000))
    UName = input('What is the User Name/Nickname? ')
    pswd = int(input('Please set your password (6 integers): '))
    Email = input('What is your email Address? ')
    Gender = input('What is your gender? ')
    Height = float(input('What is your height(m)? '))
    OWeight = float(input('What is your Original Weight(kg)? '))
    BMI = OWeight/Height**2
    
    print('Account ID:', acct)
    print('Password:', pswd)
    print('User Name/Nickname:', UName)
    print('Email:', Email)
    print('Gender:', Gender)
    print('Height:', Height,'m')
    print('Original Weight:', OWeight, 'kg')
    print('BMI:', OWeight/Height**2)
    BMI_Refer()
    
    if BMI < 18.5 or BMI == 18.5:
        print('You are too skinny, no need to lose weight! Just work out for fun!')
        AimW = float(input('What is your aim weight? '))
        AimT = float(input('How long do you want to acchive your aim weight(In Months, e.g. 3.5)? '))
        LoseW_w = 7*(OWeight - AimW)/(AimT*30)
    if BMI > 18.5 and BMI < 24.9:
        print('You are the normal weight.')
        AimW = float(input('What is your aim weight? '))
        AimT = float(input('How long do you want to acchive your aim weight(Unit: Month, e.g. 3.5)? '))
        LoseW_w = 7*(OWeight - AimW)/(AimT*30)
    if BMI > 24.9 and BMI < 29.9:
        print('You are overweight.')
        AimW = float(input('What is your aim weight? '))
        AimT = float(input('How long do you want to acchive your aim weight(Unit: Month, e.g. 3.5)? '))
        LoseW_w = 7*(OWeight - AimW)/(AimT*30)
    if BMI > 29.9:
        print('You have acchived the level of Obesity.')
        AimW = float(input('What is your aim weight? '))
        AimT = float(input('How long do you want to acchive your aim weight(Unit: Month, e.g. 3.5)? '))
        LoseW_w = 7*(OWeight - AimW)/(AimT*30)
    
    print('Your Ami Weight is:', AimW, 'kg')
    print('Your Losing Weight Duration is: ', AimT, 'months')
    print('You should lose weight: ', LoseW_w, 'kg per week.' )
    
    NewW = OWeight
    accounts.append([acct, pswd, UName, Email, Gender, Height, OWeight, NewW, BMI, AimW, AimT, LoseW_w])
    print('A thousond miles begin with a single step! Let us get started!')

accounts.append([1234, 123456, 'Tom', 'abcd@masonlive.gmu.edu', 'Male', 1.88, 90, 90, 25.46, 80, 3, 0.78])

def authenticate(acct, pswd):
    print('Reaching out to your account, just a second....')
    for a in accounts:
        if a[0] == acct and a[1] == pswd:
            return True
    return False

def UInfo(acct):
    for a in accounts:
        if a[0] == acct:
            print(a)
    

def log_records(acct, LostW, NewW):
    records.append([acct, LostW, NewW])
    print(records)

def Change_Goal(acct):
    for a in accounts:
        if a[0] == acct:
            NAimW = float(input('What is your new aim weight? '))
            a[9] = NAimW
            NAimT = float(input('What is your new losing weight duration? '))
            a[10] = NAimT 
            a[11] = 7*(a[7] - a[9])/(NAimT*30)
            print('Your new aim weight is: ', a[9], 'kg')
            print('Your new losing weight duration is:', a[10], 'months')
            print('You should lose ', a[11], 'kg per week')
 
    
def Update_Weight(acct):
    NewW = float(input('What is your current weight? '))
    for a in accounts:
        if a[0] == acct and a[7] != a[9]:
            LostW = NewW - a[7]
            if a[7] > NewW:
                a[7] = NewW
                print('Congratulations! You lost', -LostW, 'kg.', 'Hang in there! You are closer to your aim!')
                log_records(acct, LostW, NewW)
                TotalL = a[6] - a[7]
                print('You have lost ', TotalL, 'kg in total.')
            elif a[7] < NewW:
                a[7] = NewW
                print('Crying....You gain', LostW, 'kg. Please do not give up. You can do it!')
                log_records(acct, LostW, NewW)
                TotalL = a[6] - a[7]
                print('You have lost ', TotalL, 'kg in total.')
            else:
                a[7] = NewW
                print('You lost', -LostW, 'kg.', 'Hang in there! You can do it!')
                log_records(acct, LostW, NewW)
                TotalL = a[6] - a[7]
                print('You have lost ', TotalL, 'kg in total.')
        if a[0] == acct and a[7] < a[9] or a[7] == a[9]:
                a[7] = NewW
                print('Congratulations! You got it! You have acchived your goal!')
                TotalL = a[6] - a[7]
                

import matplotlib.pyplot as plt
Y = []
def LW_Curve(acct):
    for a in accounts:
        if a[0] == acct:
            Y.append(a[6])
    for a in records:
        if a[0] == acct:
            Y.append(a[2])
    plt.plot(range(0, len(Y)), Y)
    plt.xlabel('Number of Records')
    plt.ylabel('Weight: kg')
    plt.title('Losing Weight Curve')
    plt.show()

def user_menu(acct):
    cmd = ''
    while cmd != 'Exit':
        cmd = input('Please select UInfo/Change_Goal/Update_Weight/LW_Curve: ')
        if cmd == 'UInfo':
           UInfo(acct)
        if cmd == 'Change_Goal':
           Change_Goal(acct)
        if cmd == 'Update_Weight':
           Update_Weight(acct)
        if cmd == 'LW_Curve':
           LW_Curve(acct)
           
        
def old_user():
        acct = -2
        while acct != -1:
            acct = int(input('Please provide your Account ID: '))
            if acct != -1:
                pswd = int(input('Please provide password: '))
                if authenticate(acct, pswd):
                    for a in accounts:
                        if a[0] == acct:
                            print('Welcome back!', a[2])
                            user_menu(acct)
                else:
                    print('Wrong user name or password or not existing user!')

                    
def run_slim():
    sign_in = ''
    sign_in = input('Are you a new user? Yes/No: ')
    if sign_in == 'Yes':
        sign_up()
        old_user()
    else:
        old_user()
        
run_slim()   
    