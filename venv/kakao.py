


N = 5

stages = [2, 1, 2, 6, 2, 4, 3, 3]

answer = []

not_clear = [0 for i in range(0, N+1)]

for stage in stages:
    not_clear[stage-1] += 1

fail_rate = {}
stage_clear = not_clear[N]
for i in range(0, N):
    stage_clear += not_clear[N - i - 1]
    if stage_clear is not 0:
        fail_rate[N - i - 1] = not_clear[N - i - 1] / stage_clear
    else:
        fail_rate[N - i - 1] = 0

fail_rate_2 = sorted(fail_rate.items(), key=lambda kv: kv[1], reverse=True)
for i in range(0, N):
    answer.append(fail_rate_2[i][0] + 1)