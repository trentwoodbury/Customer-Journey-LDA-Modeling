import matplotlib.pyplot as plt


if __name__ == '__main__' :
    fig = plt.figure()
    fig.suptitle('Optimal Process', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    # ax.set_xlabel('xlabel')
    # ax.set_ylabel('ylabel')

    #Initial Data Box
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    t = ax.text(5, 9, "Initial Data", ha="center", va="center", size=15, bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("round", pad=0.6, )
    # ax.text(3, 8, 'Initial Data', style='normal', bbox=bb)

    #Arrow to Data Formatting
    ax.annotate("", xy=(5, 7.5), xycoords='data', xytext=(5, 8.5), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    #Build Model
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    t = ax.text(5, 7, "Prototype LDA Model", ha="center", va="center", size=15, bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("round", pad=0.6, )

    #Arrow to  Model Tweaks
    ax.annotate("", xy=(5, 5.5), xycoords='data', xytext=(5, 6.5), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    #Model Tweaks
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    t = ax.text(5, 5, "Tweak Model for Optimization", ha="center", va="center", size=15, bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("round", pad=0.6, )

    #Arrow to  interpreting the results
    ax.annotate("", xy=(5, 3.5), xycoords='data', xytext=(5, 4.5), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    #Interpreting the Results
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    t = ax.text(5, 3, "Interpret Results", ha="center", va="center", size=15, bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("round", pad=0.6, )

    #Arrow to  communicating insights
    ax.annotate("", xy=(5, 1.5), xycoords='data', xytext=(5, 2.5), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    #Communicate Results
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    t = ax.text(5, 1, "Communicate Results", ha="center", va="center", size=15, bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("round", pad=0.6, )

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis([0, 10, 0, 10])

    plt.savefig('optimal.png')
