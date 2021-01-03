import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors
import os
import pandas as pd
import numpy as np


"""
class TriView:
    def  __init__(self):
        pass
    
    @staticmethod
    def _get_xy(tensor):
        x, y, z = tensor[:,0], tensor[:,1], tensor[:,2]
        view_x = 0.5 * (x + 2 * z) / (x + y + z)
        view_y = 3**0.5 *0.5 * (x) / (x + y + z)
        return view_x, view_y

    def get_tri_graph(self, atoms, tensor, view_z):
        view_x, view_y = self._get_xy(tensor)
        T = tri.Triangulation(view_x, view_y)

        fig = plt.figure()
        plt.rcParams['font.family'] = 'Times New Roman'
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', 'datalim')
        plt.tick_params(labelbottom = False, labelleft = False, labelright = False, labeltop = False)
        plt.tick_params(bottom = False, left = False, right = False, top = False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        h = np.sqrt(3.0)*0.5

        #外周
        ax1.plot([0.0, 1.0],[0.0, 0.0], 'k-', lw = 2)
        ax1.plot([0.0, 0.5],[0.0, h], 'k-', lw = 2)
        ax1.plot([1.0, 0.5],[0.0, h], 'k-', lw = 2)

        #頂点のラベル
        ax1.text(0.5, h+0.1, atoms[0], fontsize = 16, ha = 'center', va = 'top')
        ax1.text(-0.02 - len(atoms[1]) * 0.05, -0.02, atoms[1], fontsize = 16)
        ax1.text(1.025, -0.02, atoms[2], fontsize = 16, zorder=10, linespacing = 0)

        #軸ラベル
        for i in range(1,10):
            ax1.text(0.52+(10-i)/20.0, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=10)
            ax1.text((10-i)/20.0-0.07, h*(10-i)/10.0-0.0, '%d0' % i, fontsize=10)
            ax1.text(i/10.0-0.03, -0.06, '%d0' % i, fontsize=10)

        #内側目盛
        for i in range(1,10):
            ax1.plot([i/20.0, 1.0-i/20.0],[h*i/10.0, h*i/10.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([i/20.0, i/10.0],[h*i/10.0, 0.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([0.5+i/20.0, i/10.0],[h*(1.0-i/10.0), 0.0], color='#AAAAAA', lw=0.5, zorder=2)

        # カラーバーの範囲
        vmin = np.min(view_z)
        vmax = np.max(view_z)
        norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)

        #プロット間隔
        levels = []
        while vmin <= vmax * 51 / 50:
            levels.append(float(vmin))
            vmin = float(vmin) + abs(float(vmax)) / 50
        cmap = plt.cm.rainbow

        # 図の設定
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, shrink=0.8, ticklocation = 'top')

        # プロット 
        plt.tricontourf(view_x, view_y, T.triangles, view_z, cmap = cmap, norm = norm, levels = levels, zorder=1)
        plt.scatter(view_x, view_y, c = view_z, s = 40, linewidth = 1, edgecolor = 'black', norm = norm, cmap = cmap, zorder=4)
        plt.rcParams['font.family'] = 'Times New Roman'
        return fig
        """

class TriView:
    def  __init__(self):
        pass
    
    @staticmethod
    def _get_xy(tensor):
        x, y, z = tensor[:,0], tensor[:,1], tensor[:,2]
        view_x = 0.5 * (x + 2 * z) / (x + y + z)
        view_y = 3**0.5 *0.5 * (x) / (x + y + z)
        return view_x, view_y

    def get_tri_graph(self, atoms, tensor, view_z):
        view_x, view_y = self._get_xy(tensor)
        T = tri.Triangulation(view_x, view_y)

        # fig = plt.figure(tight_layout=dict(pad=2))
        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1)
        plt.rcParams['font.family'] = 'Times New Roman'
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', 'datalim')
        ax1.tick_params(labelbottom = False, labelleft = False, labelright = False, labeltop = False)
        ax1.tick_params(bottom = False, left = False, right = False, top = False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        h = np.sqrt(3.0)*0.5

        #外周
        ax1.plot([0.0, 1.0],[0.0, 0.0], 'k-', lw = 2)
        ax1.plot([0.0, 0.5],[0.0, h], 'k-', lw = 2)
        ax1.plot([1.0, 0.5],[0.0, h], 'k-', lw = 2)

        #頂点のラベル
        ax1.text(0.5, h+0.1, atoms[0], fontsize = 16, ha = 'center', va = 'top')
        ax1.text(-0.02 - len(atoms[1]) * 0.05, -0.02, atoms[1], fontsize = 16)
        ax1.text(1.025, -0.02, atoms[2], fontsize = 16, zorder=10, linespacing = 0)

        #軸ラベル
        for i in range(1,10):
            ax1.text(0.52+(10-i)/20.0, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=10)
            ax1.text((10-i)/20.0-0.07, h*(10-i)/10.0-0.0, '%d0' % i, fontsize=10)
            ax1.text(i/10.0-0.03, -0.06, '%d0' % i, fontsize=10)

        #内側目盛
        for i in range(1,10):
            ax1.plot([i/20.0, 1.0-i/20.0],[h*i/10.0, h*i/10.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([i/20.0, i/10.0],[h*i/10.0, 0.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([0.5+i/20.0, i/10.0],[h*(1.0-i/10.0), 0.0], color='#AAAAAA', lw=0.5, zorder=2)

        # カラーバーの範囲
        vmin = np.min(view_z)
        vmax = np.max(view_z)
        norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)

        #プロット間隔
        levels = []
        while vmin <= vmax * 51 / 50:
            levels.append(float(vmin))
            vmin = float(vmin) + abs(float(vmax)) / 50
        cmap = plt.cm.rainbow

        # 図の設定
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, shrink=0.8)

        # プロット
        plt.xlim(0,1)
        ax1.tricontourf(view_x, view_y, T.triangles, view_z, cmap = cmap, norm = norm, levels = levels, zorder=1)
        # ax1.scatter(view_x, view_y, c = view_z, s = 40, linewidth = 1, edgecolor = 'black', norm = norm, cmap = cmap, zorder=4)
        plt.rcParams['font.family'] = 'Times New Roman'
        return fig
    
    def add_plot(self, graph, tensor, view_z=np.array([None])):
        view_x, view_y = self._get_xy(tensor)
        T = tri.Triangulation(view_x, view_y)
        # fig = plt.figure(tight_layout=dict(pad=2))]
        # plt.figure(tight_layout=dict(pad=2))
        ax1 = graph.add_subplot(111)
        graph.subplots_adjust(left = 0, right=1, bottom=0.2, top=1.1)
        if not np.any(view_z):
            ax1.scatter(view_x, view_y, c = "black", s = 20, linewidth = 1, zorder=4)
        else:
            # カラーバーの範囲
            vmin = np.min(view_z)
            vmax = np.max(view_z)
            norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
            cmap = plt.cm.rainbow

            # 図の設定
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            ax1.scatter(view_x, view_y, c = view_z, s = 20, linewidth = 1, edgecolor = 'black', norm = norm, cmap = cmap, zorder=4)
        return graph