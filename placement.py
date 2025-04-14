### here we determine optimal placement of turbines, based on the results of NN prediction

def is_safe(turbines, lat, long, static_lat, static_long, n):
    if long == static_long:
        return False

    for i in range(long):
        if turbines[i] == lat or abs(turbines[i] - lat) == abs(i - long):
            return False

    return True

def solve_n_turbines(turbines, long, static_lat, static_long, n):
    if long >= n:
        return True

    if long == static_long:
        return solve_n_turbines(turbines, long + 1, static_lat, static_long, n)

    for i in range(n):
        if i == static_lat and long != static_long:
            continue
        if is_safe(turbines, i, long, static_lat, static_long, n):
            turbines[long] = i
            if solve_n_turbines(turbines, long + 1, static_lat, static_long, n):
                return True
            turbines[long] = -1

    return False

def print_solution(turbines, n):
    field = [['.' for _ in range(n)] for _ in range(n)]
    for long in range(n):
        field[turbines[long]][long] = 'T'
    for lat in field:
        print(' '.join(lat))
    print()

def main():
    n = 6
    turbines = [-1] * n
    static_pos = (5, 3)
    turbines[static_pos[1]] = static_pos[0]

    if solve_n_turbines(turbines, 0, static_pos[0], static_pos[1], n):
        print_solution(turbines, n)

    ### calculation of geographical coordiates for produced by hillclimb algorithm placement
    lat_max, lat_min = 52.403834, 52.398781
    long_max, long_min = -0.936093, -0.949527
    lat_step = lat_max - lat_min
    long_step = long_max - long_min
    turbines_pos = [[0, 0],[0, 0],[0, 0],
                    [0, 0],[0, 0],[0, 0]]
    for i in range(len(turbines)):
        turbines_pos[i][0] = turbines[i]
        turbines_pos[i][1] = i

    for i in range(len(turbines)):
        turbines_pos[i][0] = (turbines_pos[i][0] * lat_step) + lat_min
        turbines_pos[i][1] = (turbines_pos[i][1] * long_step) + long_min

    for i in range(len(turbines)):
        print(f'Placement of {i+1} turbine - latitude:', round(turbines_pos[i][0], 6),
              'longitude', round(turbines_pos[i][1], 6))

if __name__ == "__main__":
    main()
