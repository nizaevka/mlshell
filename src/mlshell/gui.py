"""The module with GUI class description.

Note:
    radio 1 template is hidden in fig_element_prepare.
    text_box template is hidden in initdraw.

"""


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import seaborn as sns
from copy import deepcopy
from mlshell.libs import np, logging, pd, tabulate


# класс GUI универсальный, состав функций надо прописывать в дочернем классе
class Draw(object):
    def __init__(self):
        # Список глобальных переменных
        self.axcolor = 'lightgoldenrodyellow'

    def initdraw(self):
        fig_elements = self.fig_elements

        # Создание окна
        # Все координаты от нижнего левого угла
        self.fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # right axis y
        # self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        self.fig.set_size_inches(20, 10)  # размеры окна
        plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.3)  # нормированное(0, 1) положение графика в окне

        # Наполнение окна
        ################################################################################################################
        # Заголовок
        if 'title_text' in fig_elements:
            font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}
            self.title = plt.title(fig_elements['title_text'], fontdict=font)
            # Тестовые поля c координатами (значения ползунков)
        if 'text' in fig_elements:
            self.text_dic = {}
            iter = 0
            for i in fig_elements['text']:
                args = i[0]
                kwargs = i[1]
                self.text_dic[iter] = plt.text(*args, transform=ax1.transAxes, **kwargs)
                iter += 1
            # Графики
        if 'plot' in fig_elements:
            self.plot_dic = {}
            for key, val in fig_elements['plot'].items():
                ax = ax1  # left
                if 'right' in key:
                    ax = ax2
                self.plot_dic[key] = ax.plot(*val[0], **val[1])[0]
        if 'scatter' in fig_elements:
            self.scatter_dic = {}
            for key, val in fig_elements['scatter'].items():
                ax = ax1  # left
                if 'right' in key:
                    ax = ax2
                self.scatter_dic[key] = ax.scatter(*val[0], **val[1])
            # Кликабельный список пунктов
        if 'radio' in fig_elements:
            self.radio_dic = {}
            iter = 0
            for i in fig_elements['radio']:
                ax_radio = plt.axes(i[0], facecolor=self.axcolor)
                self.radio_dic[iter] = RadioButtons(ax_radio, i[1], active=0)
                self.radio_dic[iter].on_clicked(self.update)
                iter += 1
            # Слайдеры
        if 'slider' in fig_elements:
            self.slider_lis = []
            iter = 0
            for i in fig_elements['slider']:
                ax_slider = plt.axes(i[0], facecolor=i[1])
                self.slider_lis.append(Slider(ax_slider, i[2], i[3], i[4], valinit=i[5], valstep=1, color=i[6]))
                self.slider_lis[-1].on_changed(self.update)
                iter += 1
            # Бокс ввода текста
        if 'box' in fig_elements:
            # hide box
            self.box_val = ''
            ax_text_box = plt.axes([0, 0, 0, 0])
            # self.box_val = fig_elements['box']
            # ax_text_box = plt.axes([0.6, 0.025, 0.1, 0.04])
            self.text_box = TextBox(ax_text_box, 'SZ', initial=self.box_val)
            self.text_box.on_submit(self.tbox_val)  # при нажатии enter или если ушёл из окна

        if 'button' in fig_elements:
            # Кнопки
            resetax = plt.axes([0.80, 0.03, 0.10, 0.03])
            # hide plus/minus
            # minusax = plt.axes([0, 0, 0, 0])
            # plusax = plt.axes([0, 0, 0, 0])
            minusax = plt.axes([0.50, 0.03, 0.02, 0.03])
            plusax = plt.axes([0.55, 0.03, 0.02, 0.03])
            button = Button(resetax, 'Reset', color=self.axcolor, hovercolor='0.975')
            b_minus = Button(minusax, '-', color=self.axcolor, hovercolor='0.975')
            b_plus = Button(plusax, '+', color=self.axcolor, hovercolor='0.975')
            button.on_clicked(self.reseting)
            b_plus.on_clicked(self.plus)
            b_minus.on_clicked(self.minus)

        # color description (alternativaly can specify by axes)
        self.fig.legend(tuple(self.plot_dic.values()),
                        tuple(i._label for i in self.plot_dic.values()), 'lower left')
        self.fig.legend(tuple(self.scatter_dic.values()),
                        tuple(i._label for i in self.scatter_dic.values()), 'lower right')

        # Parameter setted only once
        self.init_param()
        # Параметры для отслеживания изменений
        self.temp_param()
        plt.show()

    def tbox_val(self, text):
        self.box_val = text
        self.update(text)

    def update(self, val):
        # считываем положение элементов = > слайдеры, боксы, радиокнопки
        self.read_val()

        # исходные данные train - test ( недоделано )
        self.radio2_handler()

        # пересчёт дискретности шага перебора
        self.textbox_handler()  # => параметры для брутфорса для данного var

        # undef
        self.radio1_handler()  # =>

        # make self.axis_y_dynamic() if need
        self.slider_handler()

        # Обновляем элементы графиков
        self.fig_element_replace()

        # обновляем temp к текущему состоянию как заключительный этап
        self.temp_param()

    def init_param(self):
        pass

    def temp_param(self):
        pass

    def read_val(self):
        pass

    def radio2_handler(self):
        pass

    def textbox_handler(self):
        pass

    def radio1_handler(self):
        pass

    def slider_handler(self):
        pass

    def fig_element_replace(self):
        pass


class GUI(Draw):
    def __init__(self, base_plot, params, logger):
        super().__init__()
        if logger is None:
            self.logger = logging.Logger('GUI')
        else:
            self.logger = logger
        self.base_plot = self.check_base_plot(base_plot)

        # Исходные данные - задаётся пользователем
        self.user_params(params)
        # Делаем равными если где-то свечки были пропущены
        self.gap_recover()

    def check_base_plot(self, base_plot):
        if isinstance(base_plot, pd.DataFrame):
            if base_plot.shape[1] > 1:
                raise ValueError("Base plot should be pd.DataFrame with one column or pd.Series")
            return base_plot.iloc[:, 0]
        elif isinstance(base_plot, pd.Series):
            return base_plot
        else:
            raise ValueError("Base plot should be pd.DataFrame with one column or pd.Series, not {}".format(type(base_plot)))

    def user_params(self, params):
        self.estimator_type = params['pipeline__type']
        self.data = params['data']
        self.train_index = params['train_index'].tolist()  # без tolist error почему-то в loc
        self.test_index = params['test_index'].tolist()
        self.estimator = params['estimator']
        self.hp_grid = params['gs__hp_grid']  # self.hp_grid = {'hp1_name':range_hp1, 'hp2_name':range_hp2, }
        self.best_params_ = params['best_params_']  # from gs  по ключу должно быть число
        self.hp_grid_flat = params['hp_grid_flat']  # self.hp_grid = {'hp1_name':range_hp1, 'hp2_name':range_hp2, }
        self.best_params_flat = params['best_params_flat']  # from gs  по ключу должно быть число
        self.metric = params['metric']
        self.box_val = ' '.join([str(val.shape[0]) for val in
                                 self.hp_grid_flat.values()])  # число шагов по разности курсов - продолжительность по времени (число точек, чтоб как-то учесть кратковременность всплесков)
        columns = self.data.columns
        self.train = self.data.loc[self.train_index]
        self.test = self.data.loc[self.test_index]
        self.X_train = self.train[[name for name in columns if 'feature' in name]]
        self.y_train = self.train['targets']
        self.X_test = self.test[[name for name in columns if 'feature' in name]]
        self.y_test = self.test['targets']

        if self.estimator_type == 'classifier':
            self.logger.info("Gui dynamic visualisation implements:\n"
                             "    precision plot, TP/FP/FN scatters on base_plot")
        elif self.estimator_type == 'regressor':
            self.logger.info("Gui dynamic visualisation implements:\n"
                             "    r2/mae/mse plots, residuals scatter on base_plot")
        else:
            raise ValueError("Unknown estimator type")

    def gap_recover(self):
        # fill the gap
        pass

    def plot(self, plot_flag=True, base_sort=False):
        # Рассчитываем данные для первого запуска графика
        ###################################################################################
        # Количество sliders: количество варьируемых параметров в GS
        # radio-button: смена train - test
        # text_box: sliders_subgrid
        # axis_x:
        #  samples index
        # asix_y:
        #  base_plot, TP_,FN_,FP_scatters, score_plot

        # Default value:
        # radio - 'validation'
        # sliders - best_hp from GS
        # text_box - 'maxsize grid for sliders'

        # isneed to sort base_plot

        # init data:  test
        if base_sort:
            self.train_index_base_sort = self.base_plot[self.train_index].sort_values(inplace=False).index
            self.test_index_base_sort = self.base_plot[self.test_index].sort_values(inplace=False).index
        else:
            self.train_index_base_sort = self.train_index
            self.test_index_base_sort = self.test_index

        self.index_current = self.test_index_base_sort

        # on change of: self.index_current
        self.current_data()  # => self.X_current, self.y_current,  self.x, self.y['base_plot']

        # on change of: self.hp_grid_flat
        self.sliders_data()  # => self.sliders_params['index', 'values', 'sort_values', 'orig_name', 'short_name', 'length']

        # prepare hp subgrid for sliders
        # on change of: self.box_val, self.hp_grid_flat
        self.sliders_subgrid()  # => self.sliders_params['subindex', 'sort_subvalues','sublength']

        # recalc best_param_ on hp subgrid
        # on change of: self.box_val, self.hp_grid_flat, self.best_param_flat
        self.best_params_subgrid()  # => self.sliders_params['best_index', 'best_val','current_index']

        # recalc y_plots dynamic related to sliders
        # on change of: self.index_current, self.sliders_params['current_index']
        self.axis_y_dynamic()  # => m

        if plot_flag:
            # Формируем элементы графика
            ###################################################################################
            Draw.__init__(self)  # чтобы прошла инициализация родительского класса
            self.fig_element_prepare()  # => params

            # Отрисовка
            ###################################################################################
            self.initdraw()
        else:
            if self.estimator_type == 'classifier':
                self.logger.info('score: {} TP: {} FP: {}'.format(round(self.y['score_plot_right'][-1], 4),
                                                                  np.count_nonzero(self.meta['TP'] != 0),
                                                                  np.count_nonzero(self.meta['FP'] != 0)))
                # np.around(self.y[3], 2) не обрубает нули
            else:
                self.logger.info('score: {} MAE: {} MSE: {}'.format(round(self.y['score_plot_right'][-1], 4),
                                                                    round(self.meta['MAE'][-1], 2),
                                                                    round(self.meta['MSE'][-1], 2)))
            # текстовые сообщения
            self.logger.info('{}'.format(self.text1_prepare(self.sliders_params)))

    def current_data(self):
        self.data_current = self.data.loc[self.index_current]
        self.base_plot_current = self.base_plot.loc[self.index_current].values

        # [deprecated] no original index
        # self.base_plot_current = self.base_plot[self.index_current]

        columns = self.data_current.columns
        self.X_current = self.data_current[[name for name in columns if 'feature' in name]]
        self.y_current = self.data_current['targets']

        self.axis_x_static()
        self.axis_y_static()

    def axis_x_static(self):
        self.x = np.arange(self.X_current.values.shape[0], dtype=np.uint64)

    def axis_y_static(self):
        self.y = {
            'base_plot': (self.y_current.values if self.estimator._estimator_type == 'regressor'
                          else self.base_plot_current),
        }

    def sliders_data(self):
        """ Prepare hp_grid to use in slider"""
        self.sliders_params = []
        if self.hp_grid:
            self.logger.info('Sliders map index-value:')
        for i, key in enumerate(self.hp_grid_flat.keys()):
            values = self.hp_grid_flat[key]
            l_ = values.shape[0]
            if not isinstance(values[0], np.object):
                # FunctionTransformer for example
                values_sorted = np.ascontiguousarray(np.sort(values))
            else:
                values_sorted = np.ascontiguousarray(values)
            self.sliders_params.append({'values': values,
                                        'sort_values': values_sorted,  # self.parami
                                        'orig_name': key,
                                        'short_name': self.short_name(key),
                                        'index': np.arange(l_),  # self.pi_index
                                        'length': l_,
                                        })
            self.logger.info('{}'.format(tabulate.tabulate(pd.DataFrame(columns=[self.short_name(key)], data=values_sorted),
                                                           headers='keys', tablefmt='psql')))
            # self.logger.info('param_{} amount of values={}'.format(key, l_))

    def sliders_subgrid(self):
        ################################################################################################################
        # get steps for sliders_params from "box"
        steps = [int(step) for step in self.box_val.split()]

        # discretisize hp grid
        grid_shape = []
        for i, val in enumerate(self.sliders_params):
            l = val['length']
            h = steps[i]
            val['subindex'] = np.array(list(range(0, l, l // h)) + [l - 1], dtype=np.int64, order='C')  # self.pi_brut
            val['sort_subvalues'] = np.ascontiguousarray(val['sort_values'][val['subindex']])  # self.pi
            val['sublength'] = val['subindex'].shape[0]
            grid_shape.append(val['sublength'])

        # [Now array auto-generated in GS ]
        # initalize y array for grid
        # self.score = np.ones(grid_shape, dtype=np.double, order='C')
        # self.TP = np.zeros(grid_shape, dtype=np.double, order='C')

    def best_params_subgrid(self):
        best_params_subgrid_flat = self.brutforce()
        for i, val in enumerate(self.sliders_params):
            best_val = best_params_subgrid_flat[val['orig_name']]
            best_index = np.where(val['values'] == best_val)[0][0]
            val['best_val'] = best_val
            val['best_index'] = best_index  # self.pi_brut[self.ii]  self.parami_def_index
            val['current_index'] = val['best_index']  # self.currnti_index

    def brutforce(self):
        """Calc scores for sliders grid"""
        # !! Этот блок надо в grid-search как и subgrid , оттуда уже на отрисовку передавать параметры, причем пусть кэширует что уже обсчитал. Гораздо логичнее будет
        # [Функционал для проведения GS, перебора на subgrid]
        # Была необходимость раньше когда возился с th_
        # self.gridsearch()

        #   with open('testing.npz','wb') as f:
        #       np.savez(f, np.asarray([0], np.uint32), self.predict_proba, self.data_current, self.p_buy, self.p_sell, self.q_buy, self.q_sell , self.p1, self.p2, self.p3, self.p4,
        #                  self.bal, self.cou, self.cou_pos, self.maker)

        #   fast_arbi.brutforce(np.asarray([0], np.uint32), self.predict_proba, self.data_current, self.p_buy, self.p_sell, self.q_buy, self.q_sell , self.p1, self.p2, self.p3, self.p4,
        #                  self.bal, self.cou, self.cou_pos, self.typ) # self.maker)  # index = [0] typ=1
        # best_subindex = np.unravel_index(self.score.argmax(), self.scores.shape)  # argmax - находит flat index, unravel перестраивает в двойной

        return self.best_params_flat
        ######################################################################################################################################

    def axis_y_dynamic(self):
        # Create unflat hp_params dictionary from current sliders_params
        sliders_hp_params = {}
        for i, val_ in enumerate(self.sliders_params):
            key = val_['orig_name']
            val = val_['sort_values'][val_['current_index']]
            if isinstance(key, tuple):
                # functiontransformer kw_args compliance
                if key[0] not in sliders_hp_params:
                    sliders_hp_params[key[0]] = {}
                sliders_hp_params[key[0]][key[1]] = val
            elif isinstance(val, np.ndarray) and len(val.shape) > 1:  # [[0,100],[25,75]] => (0,100)
                sliders_hp_params[key] = tuple(val)
            else:
                sliders_hp_params[key] = val
        current_hp_params = deepcopy(self.best_params_)
        current_hp_params.update(sliders_hp_params)
        # Make prediction on current_data
        self.estimator.set_params(**current_hp_params)
        print('Train and predict.')
        y_pred = self.estimator.fit(self.X_train, self.y_train).predict(self.X_current)
        y_true = self.y_current.values
        # y_true_score = self.y_score.values[self.index_current] deprecated
        score, meta = self.metric(y_true, y_pred, meta=True)
        self.meta = meta
        
        self.y['score_plot_right'] = meta['score']
        if self.estimator_type == 'classifier':
            self.y['TP_scatter'] = meta['TP'] * self.y['base_plot']
            self.y['FP_scatter'] = meta['FP'] * self.y['base_plot']
            self.y['FN_scatter'] = meta['FN'] * self.y['base_plot']
        else:
            bp_max = np.max(self.y['base_plot'])
            self.y['MAE_plot'] = (meta['MAE']/np.max(meta['MAE'])) * bp_max
            self.y['MSE_plot'] = (meta['MSE'] / np.max(meta['MSE'])) * bp_max
            self.y['RES_scatter'] = meta['RES'] + self.y['base_plot']
        pass

    def fig_element_prepare(self):
        # заголовок
        if self.estimator_type == 'classifier':
            title_text = 'score: {} TP: {} FP: {}'.format(round(self.y['score_plot_right'][-1], 4),
                                                          np.count_nonzero(self.meta['TP'] != 0),
                                                          np.count_nonzero(self.meta['FP'] != 0))
                                                          # np.around(self.y[3], 2) не обрубает нули
        else:
            title_text = 'score: {} MAE: {} MSE: {}'.format(round(self.y['score_plot_right'][-1], 4),
                                                            round(self.meta['MAE'][-1], 2),
                                                            round(self.meta['MSE'][-1], 2))
                                                           # np.around(self.y[3], 2) не обрубает нули
        # текстовые сообщения
        # https://matplotlib.org/3.1.1/tutorials/text/text_props.html
        kwargs = {'va': 'top', 'family': 'monospace'}  # 'va': 'center'
        text0 = [(0.75, 0.95, ''), kwargs]
        text1 = [(0.05, 0.95, '{}'.format(self.text1_prepare(self.sliders_params))), kwargs]

        plots = {}
        scatters = {}
        if self.estimator_type == 'classifier':
            colors = {'TP_scatter': 'green', 'FP_scatter': 'red', 'FN_scatter': 'blue', 'score_plot_right': 'blue',
                      'base_plot': 'black'}
        else:
            colors = {'RES_scatter': 'red', 'MAE_plot': 'green', 'MSE_plot': 'grey', 'score_plot_right': 'blue',
                      'base_plot': 'black'}
        for key, val in self.y.items():
            if key not in colors:
                colors[key] = None
            if 'plot' in key:
                args = (self.x, val, colors[key])
                kwargs = {'label': key}
                plots[key] = (args, kwargs)
            elif 'scatter' in key:
                args = (self.x[val > 0], val[val > 0], 35, colors[key])
                kwargs = {'label': key}
                scatters[key] = (args, kwargs)

        # radio buttons
        list0 = [' '] + [' ']  # берется нулевой из списка
        list1 = ['validation', 'train']
        # hide radio0
        radio0 = ([0, 0, 0, 0], list0)  # x,y,dx,dy
        # radio0 = ([0.025, 0., 0.15, 0.27], list0)  # x,y,dx,dy
        radio1 = ([0.175, 0.01, 0.1, 0.075], list1)
        # slider
        sliders = []
        y1 = iter(range(10, 30, 20 // len(self.sliders_params))) if len(self.sliders_params) != 0 else None
        for val in self.sliders_params:
            slider = (
                [0.45, y1.__next__() / 100., 0.5, 0.01], self.axcolor, val['short_name'], val['index'][0],
                val['index'][-1],
                val['current_index'], 'blue')
            sliders.append(slider)

        # box ввода текста
        box = self.box_val

        self.fig_elements = {'title_text': title_text,
                             'text': [text0, text1, ],
                             'plot': plots,
                             'scatter': scatters,
                             'radio': [radio0, radio1, ],
                             'slider': sliders,
                             'box': box,
                             'button': True,
                             }

    # Вызываются из функции update
    ##############################################################################################
    def reseting(self, event):
        for i, val in enumerate(self.slider_lis):
            val.valinit = self.sliders_params[i]['best_index']
            val.reset()  # => set_val(valinit) => update

    def plus(self, event):
        # last clicked slider only
        ind = self.slider_index_last  # 0
        val = self.sliders_params[ind]
        val['current_index'] = min(val['index'][-1], val['current_index'] + 1)
        while val['current_index'] != val['index'][-1]:
            val['current_index'] = min(val['index'][-1], val['current_index'] + 1)
        self.slider_lis[ind].set_val(val['current_index'])

    def minus(self, event):
        # last clicked slider only
        ind = self.slider_index_last  # 0
        val = self.sliders_params[ind]
        val['current_index'] = max(0, val['current_index'] - 1)
        while val['current_index'] != 0:
            val['current_index'] = max(0, val['current_index'] - 1)
        self.slider_lis[ind].set_val(val['current_index'])

    def init_param(self):
        self.slider_index_last = 0

    def temp_param(self):
        self.current_rad1_temp = self.fig_elements['radio'][0][1][0]
        self.current_rad2_temp = self.fig_elements['radio'][1][1][0]
        for i, val in enumerate(self.sliders_params):
            val['current_index_temp'] = val['current_index']
        self.box_val_temp = self.box_val

    def read_val(self):
        self.current_rad1 = str(self.radio_dic[0].value_selected)
        self.current_rad2 = str(self.radio_dic[1].value_selected)
        for i, val in enumerate(self.sliders_params):
            val['current_index'] = int(self.slider_lis[i].val)
        self.text_dic[0].set_text('')

    # Заготовка
    def radio2_handler(self, force_flag=False):
        # print(f'radio2_handler: current={self.current_rad2} temp={self.current_rad2_temp}')
        if self.current_rad2 != self.current_rad2_temp or force_flag:
            if self.current_rad2 == 'validation':
                self.index_current = self.test_index_base_sort
            else:
                self.index_current = self.train_index_base_sort
            self.current_data()
            self.axis_x_static()
            self.axis_y_static()
            self.axis_y_dynamic()
            for key, val in self.plot_dic.items():
                val.set_xdata(self.x)
                val.set_ydata(self.y[key])
            for key, val in self.scatter_dic.items():
                val.set_offsets(np.c_[self.x, self.y[key]])
            self.fig_elements['radio'][1][1][0] = self.current_rad2
            # do in temp_param
            # self.current_rad2_temp = self.current_rad2
            # self.slider_set_best()  # => update, would second predict self.axis_y_dynamic()

    def textbox_handler(self, force_flag=False):
        # [deprecated] all results come from GS, otherwise be carefull with threshold strategy
        # print(f'textbox_handler: current={self.box_val} temp={self.box_val_temp}')
        if self.box_val != self.box_val_temp or force_flag:
            # self.logger.info(self.box_val)
            self.sliders_subgrid()
            self.best_params_subgrid()
            self.axis_y_dynamic()
            for key, val in self.plot_dic.items():
                val.set_xdata(self.x)
                val.set_ydata(self.y[key])
            for key, val in self.scatter_dic.items():
                val.set_offsets(np.c_[self.x, self.y[key]])
            # self.box_val_temp = self.box_val # do in temp_param
            self.slider_set_best()

    def radio1_handler(self, force_flag=False):
        # [Заготовка]
        # print(f'radio1_handler: current={self.current_rad1_temp} temp={self.current_rad1_temp}')
        if self.current_rad1 != self.current_rad1_temp or force_flag:
            # self.logger.info('recalc_rad1')
            self.text_dic[0].set_text('Ready')
            # do in temp_param
            # self.current_rad1_temp = self.current_rad1
            self.fig_elements['radio'][0][1][0] = self.current_rad1

    def slider_handler(self):
        needrecalc = False
        for i, val in enumerate(self.sliders_params):
            if val['current_index'] != val['current_index_temp']:
                needrecalc = True
                self.slider_index_last = i
        # print(f'slider_handler needrecalc: {needrecalc}')
        if needrecalc:
            self.axis_y_dynamic()

    def slider_set_best(self):
        for i, val in enumerate(self.sliders_params):
            if val['current_index'] != val['best_index']:
                self.slider_lis[i].set_val(val['best_index'])  # => вызывает update(val)

    def fig_element_replace(self):
        # обновляем данные графиков
        # оси тоже надо обновить

        for key, val in self.plot_dic.items():
            val.set_ydata(self.y[key])
        for key, val in self.scatter_dic.items():
            val.set_offsets(np.c_[self.x, self.y[key]])

        if self.estimator_type == 'classifier':
            self.title.set_text('score: {} TP: {} FP: {}'.format(round(self.y['score_plot_right'][-1], 4),
                                                                 np.count_nonzero(self.meta['TP'] != 0),
                                                                 np.count_nonzero(self.meta['FP'] != 0)))
                                                                # np.around(self.y[3], 2) не обрубает нули
        else:
            self.title.set_text('score: {} MAE: {} MSE: {}'.format(round(self.y['score_plot_right'][-1], 4),
                                                                   round(self.meta['MAE'][-1], 2),
                                                                   round(self.meta['MSE'][-1], 2)))
                                                                # np.around(self.y[3], 2) не обрубает нули

        self.text_dic[1].set_text('{}'.format(self.text1_prepare(self.sliders_params)))

        # update axes
        for ax in self.fig.axes:
            # recompute the ax.dataLim
            # TODO: for collections
            # https://github.com/matplotlib/matplotlib/issues/7413 https://stackoverflow.com/questions/51323505/how-to-make-relim-and-autoscale-in-a-scatter-plot
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()

        # перерисовыаем
        self.fig.canvas.draw_idle()

    def short_name(self, long):
        if isinstance(long, tuple):
            # functiontransformer kw_args compliance
            long = '__'.join(long)
        lis = long.split('__')
        if len(lis) > 2:
            short = '__'.join(lis[-2:])
        else:
            short = long
        return short

    def text1_prepare(self, sliders_params):
        lis = []
        for val in sliders_params:
            # only two last hp path
            name = val['short_name']
            value = val['values'][val['current_index']]
            if np.issubdtype(type(value), np.inexact):
                value_ = f'{value:.3f}'
            elif np.issubdtype(type(value), np.object_):
                value_ = f"too long, see table index={val['current_index']}"
            else:
                value_ = value
            #  if np.issubdtype(value, np.number) else value
            lis.append((name, value_))
        text = '{}'.format('\n'.join(map(str, lis)))
        return text


if __name__ == '__main__':
    pass
