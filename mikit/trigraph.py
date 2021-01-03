import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors
import os
import pandas as pd
import numpy as np



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
        ax1.text(0.5 - len(atoms[0]) * 0.02, h+0.03, atoms[0], fontsize = 16)
        ax1.text(-0.02 - len(atoms[1]) * 0.05, -0.02, atoms[1], fontsize = 16)
        ax1.text(1.025, -0.02, atoms[2], fontsize = 16)

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
        fig.colorbar(sm, shrink=0.8, pad = 0.15)

        # プロット 
        plt.tricontourf(view_x, view_y, T.triangles, view_z, cmap = cmap, norm = norm, levels = levels, zorder=1)
        # plt.scatter(view_x, view_y, c = view_z, s = 40, linewidth = 2, edgecolor = 'black', norm = norm, cmap = cmap, zorder=4)
        plt.rcParams['font.family'] = 'Times New Roman'
        return fig



        # キャリアのインデックス
        # conductor_idx = np.where(self.ele==self.conductor)[0][0]


'''

# パス
path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"


class Graph:
    def  __init__(self, conductor, tensor):
        # キャリア定義
        if conductor == "Oxide":
            path_element = path_root + 'input/Oxide_Propety.csv'
            self.conductor = "O"
        elif conductor == "Fluoride":
            path_element = path_root + 'input/Fluoride_Propety.csv'
            self.conductor = "F"

        self.tensor = tensor
        self.df_atoms = pd.read_csv(path_element, header = 0, index_col = 0)
        self.ele = self.df_atoms.index.values

        # キャリアのインデックス
        # conductor_idx = np.where(self.ele==self.conductor)[0][0]




        
    def Triangle(self, comp3_list, columns, path_csv):
        # csv読み込み
        df_comps = pd.read_csv(path_csv, header = 0, index_col = 0)
        
        # 組成比の追加
        # sum_comp = np.array([np.sum(self.df_atoms["comp"].values * t) for t in self.tensor])
        # sum_ionrad = np.array([np.sum(self.df_atoms["ion_rad"].values * t) for t in self.tensor])
        # print(sum_ionrad)

        # 三成分のテンソルとデータフレーム 
        comp3_idx = np.array([np.where(self.ele == c)[0][0] for c in comp3_list])
        comp3_tensor = self.tensor[:, comp3_idx]
        # print(comp3_tensor)
        comp3_df = self.df_atoms.iloc[comp3_idx]
        
        # conductorのテンソル
        conductor_idx = np.where(self.ele==self.conductor)[0][0]
        conductor_tensor = self.tensor[:, conductor_idx]
        conductor_df = self.df_atoms.iloc[conductor_idx]

        # 組成比の追加
        sum_comp_ratio = np.array([np.sum(comp3_df["comp"].values * t) for t in comp3_tensor])
        conductor_ratio = np.array([np.sum(conductor_df["comp"] * t) for t in conductor_tensor])
        # print(conductor_ratio[:7].shape)
        # print(sum_comp_ratio[:7].shape)
        # print(df_comps.values.shape)
        # print(df_comps)
        # print(conductor_ratio[:7] / sum_comp_ratio[:7])
        df_comps["comp_ratio"] = conductor_ratio[:7] / sum_comp_ratio[:7]

        # イオン半径比の追加
        # print(comp3_df["ion_rad"].values)
        sum_rad_ratio = np.array([np.sum(comp3_df["ion_rad"].values * t) for t in comp3_tensor])
        # conductor_rad = np.array([np.sum(conductor_df["ion_rad"] * 1) for t in np.ones_like(conductor_tensor)])
        conductor_rad = np.array([np.sum(conductor_df["ion_rad"] * t) for t in conductor_tensor])
        df_comps["ionrad_ratio"] = sum_rad_ratio[:7] / conductor_rad[:7]
        # print(conductor_tensor)
        if columns == "ionrad_ratio":
            print(conductor_rad)
            print(sum_rad_ratio)

        # df_compsの保存
        df_comps.to_csv("3comp_new.csv")

        # X, Y, Z, T
        for l in range(len(comp3_tensor)):
            comp3_tensor[l] /= comp3_tensor.sum(axis = 1)[l]
        X = (comp3_tensor[:, 0] + 2 * comp3_tensor[:, 2]) / 2
        Y = (3 ** (0.5) * comp3_tensor[:, 0]) / 2
        Z = df_comps[columns].values
        X = X[:7]
        Y = Y[:7]
        Z = Z[:7]
        T = tri.Triangulation(X, Y)

        # 最大の組成
        # max_idx = y.index(max(y))
        # max_comp = x[max_idx]
        # max_sigma = str(round(max(y), 2))

        #グラフの作成
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title(columns, loc = "center", pad = -0.15)
        # fig.suptitle(columns)
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
        # -0.12 2
        # -0.07 1
        ax1.text(0.5 - len(comp3_list[0]) * 0.02, h+0.03, comp3_list[0], fontsize = 16)
        ax1.text(-0.02 - len(comp3_list[1]) * 0.05, -0.02, comp3_list[1], fontsize = 16)
        ax1.text(1.025, -0.02, comp3_list[2], fontsize = 16)

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
        vmin = np.min(Z)
        vmax = np.max(Z)
        norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)

        #プロット間隔
        levels = []
        while vmin <= vmax * 51 / 50:
            levels.append(float(vmin))
            vmin = float(vmin) + abs(float(vmax)) / 50
        if columns == "rate":
            cmap = plt.cm.rainbow_r
        else:
            cmap = plt.cm.rainbow

        # 図の設定
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, shrink=0.8, pad = 0.15)

        # プロット 
        plt.tricontourf(X, Y, T.triangles, Z, cmap = cmap, norm = norm, levels = levels, zorder=1)
        plt.scatter(X, Y, c = Z, s = 40, linewidth = 2, edgecolor = 'black', norm = norm, cmap = cmap, zorder=4)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.savefig(columns + ".pdf")
        plt.clf()

    def BlackTriangle(self, comp3_list, path_csv):
        # csv読み込み
        df_comps = pd.read_csv(path_csv, header = 0, index_col = 0)
        columns = "Probability"
        
        # 組成比の追加
        # sum_comp = np.array([np.sum(self.df_atoms["comp"].values * t) for t in self.tensor])
        # sum_ionrad = np.array([np.sum(self.df_atoms["ion_rad"].values * t) for t in self.tensor])
        # print(sum_ionrad)

        # 三成分のテンソルとデータフレーム 
        comp3_idx = np.array([np.where(self.ele == c)[0][0] for c in comp3_list])
        comp3_tensor = self.tensor[:, comp3_idx]
        comp3_df = self.df_atoms.iloc[comp3_idx]
        
        # conductorのテンソル
        conductor_idx = np.where(self.ele==self.conductor)[0][0]
        conductor_tensor = self.tensor[:, conductor_idx]
        conductor_df = self.df_atoms.iloc[conductor_idx]

        # 組成比の追加
        sum_comp_ratio = np.array([np.sum(comp3_df["comp"].values * t) for t in comp3_tensor])
        conductor_ratio = np.array([np.sum(conductor_df["comp"] * t) for t in conductor_tensor])
        df_comps["comp_ratio"] = conductor_ratio / sum_comp_ratio

        # イオン半径比の追加
        sum_rad_ratio = np.array([np.sum(comp3_df["ion_rad"].values * t) for t in comp3_tensor])
        conductor_rad = np.array([np.sum(conductor_df["ion_rad"] * 1) for t in np.ones_like(conductor_tensor)])
        print(conductor_rad)
        df_comps["ionrad_ratio"] = sum_rad_ratio / conductor_rad

        # df_compsの保存
        df_comps.to_csv("3comp_new.csv")

        # X, Y, Z, T
        for l in range(len(comp3_tensor)):
            comp3_tensor[l] /= comp3_tensor.sum(axis = 1)[l]
        X = (comp3_tensor[:, 0] + 2 * comp3_tensor[:, 2]) / 2
        Y = (3 ** (0.5) * comp3_tensor[:, 0]) / 2
        Z = df_comps[columns].values
        T = tri.Triangulation(X, Y)

        # 最大の組成
        # max_idx = y.index(max(y))
        # max_comp = x[max_idx]
        # max_sigma = str(round(max(y), 2))

        #グラフの作成
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title("", loc = "center", pad = -0.15)
        # fig.suptitle(columns)
        plt.rcParams['font.family'] = 'Times New Roman'
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
        # -0.12 2
        # -0.07 1
        ax1.text(0.5 - len(comp3_list[0]) * 0.02, h+0.05, comp3_list[0], fontsize = 16)
        ax1.text(-0.0 - len(comp3_list[1]) * 0.05, -0.02, comp3_list[1], fontsize = 16)
        ax1.text(1.04, -0.02, comp3_list[2], fontsize = 16)

        #軸ラベル
        for i in range(1,10):
            ax1.text(0.52+(10-i)/20.0, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=10)
            ax1.text((10-i)/20.0-0.07, h*(10-i)/10.0-0.0, '%d0' % i, fontsize=10)
            ax1.text(i/10.0-0.025, -0.06, '%d0' % i, fontsize=10)

        #内側目盛
        for i in range(1,10):
            ax1.plot([i/20.0, 1.0-i/20.0],[h*i/10.0, h*i/10.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([i/20.0, i/10.0],[h*i/10.0, 0.0], color='#AAAAAA', lw=0.5, zorder=2)
            ax1.plot([0.5+i/20.0, i/10.0],[h*(1.0-i/10.0), 0.0], color='#AAAAAA', lw=0.5, zorder=2)

        # プロット 
        # marker = o
        plt.scatter(X[:7], Y[:7], c = df_comps["dot"][:7], s = 150, marker = "o", linewidth = 0, edgecolor = 'black')
        plt.scatter(X[7:], Y[7:], c = df_comps["dot"][7:], s = 150, marker = "D", linewidth = 0, edgecolor = 'black')
        plt.savefig("BlackTri.pdf")
        plt.clf()
'''