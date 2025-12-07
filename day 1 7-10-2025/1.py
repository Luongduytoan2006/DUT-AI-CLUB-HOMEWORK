def calc_cube_volume(a):
    return a**3
def calc_rectangular_prism_volume(a,b,c):
    return a*b*c
def calc_cylinder_volume(r,h):
    return 3.14*r**2*h
def calc_sphere_volume(r):
    return 4/3*3.14*r**3

def switch_choice(choice):
    print("\n"+"-"*20+"\n")
    if choice == 1:
        a = int(input("Nhập cạnh hình lập phương: "))
        print(f"Thể tích hình lập phương là: {calc_cube_volume(a)}")
    elif choice == 2:
        b = int(input("Nhập chiều dài hình hộp chữ nhật: "))
        c = int(input("Nhập chiều rộng hình hộp chữ nhật: "))
        d = int(input("Nhập chiều cao hình hộp chữ nhật: "))
        print(f"Thể tích hình hộp chữ nhật là: {calc_rectangular_prism_volume(b,c,d)}")
    elif choice == 3:
        r = int(input("Nhập bán kính hình trụ: "))
        h = int(input("Nhập chiều cao hình trụ: "))
        print(f"Thể tích hình trụ là: {calc_cylinder_volume(r,h)}")
    elif choice == 4:
        r = int(input("Nhập bán kính hình cầu: "))
        print(f"Thể tích hình cầu là: {calc_sphere_volume(r)}")
    print("\n"+"-"*20+"\n")

while 1:
    print("Chọn hình bạn muốn tính thể tích:")
    print("1. Hình lập phương")
    print("2. Hình hộp chữ nhật")
    print("3. Hình trụ")
    print("4. Hình cầu")
    print("0. Thoát")
    choice = int(input("Nhập lựa chọn của bạn (1-4): "))
    if choice not in [0,1,2,3,4]:
        print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
        continue
    elif choice == 0:
        print("Thoát chương trình.")
        break
    else:
        switch_choice(choice)


