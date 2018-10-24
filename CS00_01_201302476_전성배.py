month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
date = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
day = ["일요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일"]

print("2018년에서 월,일을 입력하면 무슨 요일인지 알려주는 함수를 작성하시오 (2018년 1월 1일은 월 요일)")
print("Month")
inputMonth = int(input())

print("Date")
inputDate = int(input())

total = 0
# index에 맞춰 -1
for i in range(inputMonth - 1):
    total += date[i]
total += inputDate

# 요일출력
print(day[total % 7])