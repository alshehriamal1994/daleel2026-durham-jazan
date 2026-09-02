# Which domain decided the shared task? (paper, Appendix E)
# Uses only the organisers' official final ranking sheet (distinct-team rows,
# trained on both domains; the organiser baselines are excluded).
# Post-submission analysis, CPU only.
import numpy as np

T1_CLOSED = [
    ("Salah Abdo", .5191, .7589, .7117), ("BME-Daleel", .6239, .7778, .7117),
    ("OliveTrees", .5188, .8239, .6882), ("Nu_Analytics", .4751, .7418, .6714),
    ("Durham-Jazan", .5008, .7411, .6568), ("ttlab", .5560, .6864, .6451),
    ("AIDAL-ARG", .4704, .6812, .5992), ("Northwestren_12", .4366, .5622, .5648),
    ("Burhan", .4563, .5537, .5597), ("solodiscourse", .3838, .4943, .5265),
]
T2_CLOSED = [
    ("BME-Daleel", .6653, .7941, .7547), ("Nu_Analytics", .6417, .7931, .7523),
    ("ttlab", .6356, .7784, .7374), ("ImpactAi", .6587, .7615, .7323),
    ("Salah Abdo", .6838, .7481, .7316), ("Durham-Jazan", .6097, .7621, .7292),
    ("OliveTrees", .6297, .7599, .7236), ("AIDAL-ARG", .6240, .7082, .6832),
]

for name, rows in [("Task 1 Closed", T1_CLOSED), ("Task 2 Closed", T2_CLOSED)]:
    ed = np.array([r[1] for r in rows])
    db = np.array([r[2] for r in rows])
    ov = np.array([r[3] for r in rows])
    k = len(rows) // 2
    print(f"\n{name} ({len(rows)} teams)")
    print(f"  editorial sd {ed.std(ddof=1):.3f}   debate sd {db.std(ddof=1):.3f}")
    print(f"  corr with overall: editorial {np.corrcoef(ed, ov)[0,1]:+.2f}, "
          f"debate {np.corrcoef(db, ov)[0,1]:+.2f}")
    print(f"  top {k} only:      editorial {np.corrcoef(ed[:k], ov[:k])[0,1]:+.2f}, "
          f"debate {np.corrcoef(db[:k], ov[:k])[0,1]:+.2f}")
