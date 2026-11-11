%% make_figures.m  —  CALM-ECG manuscript figures (IEEE JBHI)
% Reads tidy source data written by:  python scripts/export_figure_data.py
% Writes vector PDF + 600-dpi PNG for Fig. 2–7 into figures/out/.
% Base MATLAB only (no toolboxes). Tested API level: R2021a or newer
% (tiledlayout, exportgraphics, yline, VariableNamingRule, Legend.Layout).
% Fig. 1 (study-design schematic) is drawn separately in a vector editor.
% Author: Hoang Nhut Huynh (National Chung Hsing University, Taichung, Taiwan).
clear; close all;

here = fileparts(mfilename('fullpath'));
D = fullfile(here, '..', 'results', 'figure_data');
O = fullfile(here, 'out');
if ~exist(O, 'dir'), mkdir(O); end
rd = @(f) readtable(fullfile(D, f), 'TextType', 'string', 'VariableNamingRule', 'preserve');

% Okabe–Ito palette (colour-blind safe); every series also has its own marker
C.blue = [0 114 178]/255; C.verm = [213 94 0]/255; C.green = [0 158 115]/255;
C.purple = [204 121 167]/255; C.gray = [0.45 0.45 0.45];
VAR  = ["ERM", "DRO", "DRO+SupCon"];
VCOL = {C.blue, C.verm, C.green};
VMK  = {'o', 's', '^'};
FOLDS = ["PTB-XL", "Chapman", "CODE-15%"];
LAB   = ["AF", "1dAVb", "RBBB", "LBBB", "SB", "STach"];
W1 = 3.5; W2 = 7.16;                       % IEEE one-/two-column width (in)

%% Fig. 2 — LODO gap: (a) source-val -> held-out test, (b) per-label test AUROC (ERM)
T = rd('fig2a_lodo.csv');
f = newfig(W2, 2.6);
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
off = [-0.22 0 0.22]; h = gobjects(1, 3); lo = 1;
for i = 1:3
    for v = 1:3
        s = T(T.fold == FOLDS(i) & T.variant == VAR(v), :);
        x = i + off(v);
        mv = mean(s.val_macro_auroc); mt = mean(s.test_macro_auroc);
        plot(ax, [x x], [mv mt], '-', 'Color', VCOL{v}, 'LineWidth', 0.8);
        plot(ax, x, mv, VMK{v}, 'MarkerSize', 4, 'Color', VCOL{v}, ...
             'MarkerFaceColor', 'w', 'LineWidth', 0.8);
        h(v) = errorbar(ax, x, mt, std(s.test_macro_auroc), VMK{v}, 'MarkerSize', 4, ...
             'Color', VCOL{v}, 'MarkerFaceColor', VCOL{v}, 'LineWidth', 0.8, 'CapSize', 2);
        lo = min([lo, mt - std(s.test_macro_auroc)]);
    end
end
hv = plot(ax, NaN, NaN, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'w');
ht = plot(ax, NaN, NaN, 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
xlim(ax, [0.5 3.5]); ylim(ax, [floor(lo*100)/100, 1.0]);   % zoomed axis: stated in caption
set(ax, 'XTick', 1:3, 'XTickLabel', FOLDS);
xlabel(ax, 'Held-out cohort'); ylabel(ax, 'Macro-AUROC');
legend(ax, [h hv ht], [VAR, "Source val", "Held-out test (mean \pm SD)"], ...
       'Location', 'southwest', 'Box', 'off', 'NumColumns', 1);
paneltitle(ax, '(a)');

ax = nexttile(tl);
P = rd('fig2b_perclass_auroc.csv');
M = zeros(3, 6);
for i = 1:3
    for j = 1:6
        M(i, j) = mean(P.auroc(P.fold == FOLDS(i) & P.label == LAB(j)));
    end
end
imagesc(ax, M); colormap(ax, blues(256));
cl = [floor(min(M(:))*100)/100, 1]; caxis(ax, cl);
for i = 1:3
    for j = 1:6
        tc = 'k'; if M(i, j) > cl(1) + 0.6*diff(cl), tc = 'w'; end
        text(ax, j, i, sprintf('%.3f', M(i, j)), 'HorizontalAlignment', 'center', ...
             'FontSize', 7, 'Color', tc);
    end
end
set(ax, 'XTick', 1:6, 'XTickLabel', LAB, 'YTick', 1:3, 'YTickLabel', FOLDS, ...
        'TickLength', [0 0], 'Box', 'on');
cb = colorbar(ax); cb.Label.String = 'Test AUROC (ERM, mean of 3 seeds)'; cb.FontSize = 7;
paneltitle(ax, '(b)');
savefig2(f, O, 'Fig2_lodo_gap');

%% Fig. 3 — H1 forest plot: Δ worst-group AUROC (variant − ERM), paired bootstrap 95% CI
H = rd('fig3_h1_forest.csv');
f = newfig(W1, 2.6); ax = axes(f); hold(ax, 'on');
fr = ["PTB-XL", "Chapman", "CODE-15%", "Pooled"];
y = 0; yt = []; ytl = strings(0);
for v = ["DRO", "DRO+SupCon"]
    col = C.verm; if v == "DRO+SupCon", col = C.green; end
    ytop = y;                                   % row above this variant's block
    for k = 1:4
        r = H(H.variant == v & H.fold == fr(k), :);
        y = y + 1;
        plot(ax, 100*[r.lo r.hi], [y y], '-', 'Color', col, 'LineWidth', 1.0);
        if fr(k) == "Pooled"
            plot(ax, 100*r.delta, y, 'd', 'MarkerSize', 6, 'Color', col, 'MarkerFaceColor', col);
        else
            plot(ax, 100*r.delta, y, 'o', 'MarkerSize', 4, 'Color', col, 'MarkerFaceColor', 'w');
        end
        yt(end+1) = y; ytl(end+1) = fr(k); %#ok<SAGROW>
    end
    text(ax, -3.1, ytop + 0.45, v, 'FontSize', 7.5, 'FontWeight', 'bold', 'Color', col, ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    y = y + 0.9;
end
xline(ax, 0, 'k-', 'LineWidth', 0.6);
set(ax, 'YTick', yt, 'YTickLabel', ytl, 'YDir', 'reverse', 'YLim', [0.3 y]);
xlabel(ax, '\Delta worst-group AUROC vs. ERM (pp)');
text(ax, 0.02, 0.015, '\leftarrow worse than ERM', 'Units', 'normalized', 'FontSize', 6.5, ...
     'Color', C.gray, 'VerticalAlignment', 'bottom');
xlim(ax, [-3.2 3.2]);
savefig2(f, O, 'Fig3_h1_forest');

%% Fig. 4 — Label-conditional Mondrian conformal: positive coverage by group (ERM)
Q = rd('fig4_conformal_groups.csv');
f = newfig(W2, 2.5);
tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    s = Q(Q.fold == FOLDS(i) & Q.variant == "ERM", :);
    lbl = strings(1, 8); hp = gobjects(1, 2);
    for g = 0:7
        r = s(s.group_id == g, :);
        if isempty(r), continue; end
        female = g >= 4;
        col = C.blue; mk = 'o'; if female, col = C.verm; mk = 'd'; end
        hp(1 + female) = errorbar(ax, g + 1, mean(r.positive_coverage, 'omitnan'), ...
            std(r.positive_coverage, 'omitnan'), mk, 'Color', col, 'MarkerFaceColor', col, ...
            'MarkerSize', 4, 'LineWidth', 0.8, 'CapSize', 2); %#ok<SAGROW>
        ha = plot(ax, g + 1, mean(r.coverage), 'x', 'Color', C.gray, 'MarkerSize', 5);
        lbl(g + 1) = strrep(r.group(1), ">=", "\geq") + " (" + r.n_pos_cells(1) + ")";
    end
    yline(ax, 0.90, '--k', 'target 0.90', 'FontSize', 6.5, 'LineWidth', 0.6, ...
          'LabelHorizontalAlignment', 'left');
    set(ax, 'XTick', 1:8, 'XTickLabel', lbl, 'XLim', [0.4 8.6], 'YLim', [0.3 1.0]);
    xtickangle(ax, 45);
    if i == 1, ylabel(ax, 'Empirical coverage'); end
    title(ax, "Held-out: " + FOLDS(i), 'FontWeight', 'normal', 'FontSize', 8);
end
lg = legend(ax, [hp(1) hp(2) ha], ["Positive cells, male", "Positive cells, female", ...
            "All cells"], 'Orientation', 'horizontal', 'Box', 'off');
lg.Layout.Tile = 'south';
savefig2(f, O, 'Fig4_conformal_groups');

%% Fig. 5 — Label-definition shift
f = newfig(W2, 2.6);
tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
% (a) prevalence by site (Chapman split into its two hospitals)
A = rd('fig5a_prevalence.csv');
sites = ["PTB-XL", "Shaoxing", "Ningbo", "CODE-15%", "CODE-test"];
ser = ["SB", "AF", "AFL (CinC count)"];
X = NaN(numel(sites), numel(ser));
for i = 1:numel(sites)
    for j = 1:numel(ser)
        r = A(A.site == sites(i) & A.label == ser(j), :);
        if ~isempty(r), X(i, j) = 100*r.prevalence(1); end
    end
end
ax = nexttile(tl);
b = bar(ax, X, 'grouped', 'BarWidth', 0.85, 'LineWidth', 0.4);
bc = {C.blue, C.verm, C.green};
for j = 1:3, b(j).FaceColor = bc{j}; end
set(ax, 'XTickLabel', sites, 'YLim', [0 52]); xtickangle(ax, 30);
ylabel(ax, 'Prevalence (%)');
lg = legend(ax, ["SB", "AF", "AFL"], 'Location', 'north', 'Box', 'off', 'NumColumns', 3);
lg.ItemTokenSize = [8 6];   % default token (30 pt) made the legend wider than the panel
paneltitle(ax, '(a)');

% (b) AF AUROC on the Chapman fold under three scorings (ERM, mean ± SD, 3 seeds)
B = rd('fig5b_af_afl.csv');
cols  = ["auroc_original", "auroc_shaoxing_only", "auroc_af_or_afl", "auroc_ningbo_afl_as_target"];
names = ["As harmonized", "Shaoxing only", "AF \cup AFL positive", "Ningbo AFL as target"];
ax = nexttile(tl); hold(ax, 'on');
for k = 1:4
    v = B.(cols(k));
    errorbar(ax, mean(v), k, std(v), 'horizontal', 'o', 'Color', C.blue, ...
             'MarkerFaceColor', C.blue, 'MarkerSize', 4, 'LineWidth', 0.8, 'CapSize', 2);
    text(ax, mean(v), k - 0.28, sprintf('%.3f', mean(v)), 'FontSize', 7, ...
         'HorizontalAlignment', 'center', 'Color', C.blue);   % SD < 0.002, bars invisible
end
set(ax, 'YTick', 1:4, 'YTickLabel', names, 'YDir', 'reverse', 'YLim', [0.5 4.5], ...
        'XLim', [0.90 1.0]);
xlabel(ax, 'AF AUROC, Chapman held out');
paneltitle(ax, '(b)');

% (c) EM-estimated vs true target prevalence (all folds x labels)
E = rd('fig5c_em_prevalence.csv');
ax = nexttile(tl); hold(ax, 'on');
fl = 0.1;                                   % axis floor (%); EM estimates of 0 are drawn here
plot(ax, [fl 100], [fl 100], '-', 'Color', C.gray, 'LineWidth', 0.6);
fmk = {'o', 's', '^'}; fcol = {C.blue, C.verm, C.green}; hf = gobjects(1, 3);
for i = 1:3
    s = E(E.fold == FOLDS(i), :);
    xt = 100*s.prev_true; ye = 100*s.prev_em;
    zero = ye < fl; ye(zero) = fl;
    hf(i) = plot(ax, xt(~zero), ye(~zero), fmk{i}, 'Color', fcol{i}, ...
                 'MarkerFaceColor', fcol{i}, 'MarkerSize', 4);
    plot(ax, xt(zero), ye(zero), 'v', 'Color', fcol{i}, 'MarkerSize', 5, 'LineWidth', 0.8);
    far = find(abs(log10(max(100*s.prev_em, fl) ./ xt)) > log10(2));  % > 2-fold misses
    for q = 1:numel(far)
        j = far(q);
        if mod(q, 2) == 1
            text(ax, xt(j)*1.15, ye(j), s.label(j), 'FontSize', 6.5, 'Color', fcol{i}, ...
                 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        else
            text(ax, xt(j)*0.87, ye(j), s.label(j), 'FontSize', 6.5, 'Color', fcol{i}, ...
                 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
        end
    end
end
set(ax, 'XScale', 'log', 'YScale', 'log', 'XLim', [fl 100], 'YLim', [fl/1.2 100]);
xlabel(ax, 'True prevalence (%)'); ylabel(ax, 'EM-estimated prevalence (%)');
lg = legend(ax, hf, "Held-out " + FOLDS, 'Location', 'northwest', 'Box', 'off');
lg.ItemTokenSize = [8 6];
paneltitle(ax, '(c)');
savefig2(f, O, 'Fig5_label_definition_shift');

%% Fig. 6 — CODE-test precision–recall with reader operating points
Cv = rd('fig6_pr_curves.csv'); Op = rd('fig6_operating_points.csv');
mods = ["External (sources PTB-XL+Chapman)", "In-population (sources incl. CODE-15%)"];
mcol = {C.verm, C.blue};
opm  = ["Model external (source threshold)", "Model in-population (source threshold)"];
rdr  = ["cardiology_residents", "emergency_residents", "medical_students", "dnn"];
rnm  = ["Cardiology residents", "Emergency residents", "Medical students", "DNN (Ribeiro et al.)"];
rmk  = {'^', 's', 'd', 'p'};
f = newfig(W2, 4.6);
tl = tiledlayout(f, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
hh = gobjects(1, 8);
for j = 1:6
    ax = nexttile(tl); hold(ax, 'on');
    for m = 1:2
        s = Cv(Cv.model == mods(m) & Cv.label == LAB(j), :);
        hh(m) = plot(ax, s.recall, s.precision, '-', 'Color', mcol{m}, 'LineWidth', 1.0);
        o = Op(Op.who == opm(m) & Op.label == LAB(j), :);
        hh(2 + m) = plot(ax, o.recall, o.precision, 'o', 'Color', mcol{m}, ...
                         'MarkerFaceColor', mcol{m}, 'MarkerSize', 5);
    end
    for k = 1:4
        o = Op(Op.who == rdr(k) & Op.label == LAB(j), :);
        fc = 'w'; if k == 4, fc = C.gray; end
        hh(4 + k) = plot(ax, o.recall, o.precision, rmk{k}, 'Color', 'k', ...
                         'MarkerFaceColor', fc, 'MarkerSize', 5);
    end
    set(ax, 'XLim', [0 1.02], 'YLim', [0 1.02], 'XTick', 0:0.5:1, 'YTick', 0:0.5:1);
    title(ax, LAB(j), 'FontWeight', 'normal', 'FontSize', 8); box(ax, 'on');
end
xlabel(tl, 'Recall', 'FontSize', 8); ylabel(tl, 'Precision', 'FontSize', 8);
lg = legend(ax, hh, ["Model, external", "Model, in-population", ...
            "External @ source-val threshold", "In-population @ source-val threshold", rnm], ...
            'NumColumns', 4, 'Box', 'off', 'FontSize', 7);
lg.Layout.Tile = 'south';
savefig2(f, O, 'Fig6_codetest_pr_readers');

%% Fig. 7 — Local threshold re-selection with k labeled ECGs (CODE-test)
R = rd('fig7_local_recal.csv');
R.kn = str2double(string(R.k));            % "all" (oracle rows) -> NaN
f = newfig(W1, 2.8); ax = axes(f); hold(ax, 'on');
xo = 320;                                   % x position of the oracle markers
spec = {"code15", C.verm, "External"; "chapman", C.blue, "In-population"};
hl = gobjects(1, 2);
for m = 1:2
    s = R(R.who == "ERM " + spec{m, 1} + " (local thresholds)", :);
    s = sortrows(s, 'kn');
    band = 0.25*spec{m, 2} + 0.75;          % solid light band (keeps the PDF fully vector)
    fill(ax, [s.kn; flipud(s.kn)], [s.("p2.5"); flipud(s.("p97.5"))], band, 'EdgeColor', 'none');
end
for m = 1:2
    s = sortrows(R(R.who == "ERM " + spec{m, 1} + " (local thresholds)", :), 'kn');
    hl(m) = plot(ax, s.kn, s.macro_f1_mean, '-o', 'Color', spec{m, 2}, ...
                 'MarkerFaceColor', spec{m, 2}, 'MarkerSize', 4, 'LineWidth', 1.0);
    s0 = sortrows(R(R.who == "ERM " + spec{m, 1} + " (source thresholds)", :), 'kn');
    plot(ax, s0.kn, s0.macro_f1_mean, ':', 'Color', spec{m, 2}, 'LineWidth', 1.0);
    o = R(R.who == "ERM " + spec{m, 1} + " (ORACLE, in-sample)", :);
    plot(ax, xo, o.macro_f1_mean, 'd', 'Color', spec{m, 2}, 'MarkerSize', 6, 'LineWidth', 1.0);
end
rs = ["dnn", "cardiology_residents", "emergency_residents", "medical_students"];
rl = ["DNN", "Card. res.", "Emerg. res.", "Med. stud."];
side = {'left', 'right', 'left', 'right'};
for k = 1:4
    yv = mean(R.macro_f1_mean(R.who == rs(k)));
    yline(ax, yv, '--', rl(k), 'Color', C.gray, 'FontSize', 6, 'LineWidth', 0.5, ...
          'LabelHorizontalAlignment', side{k}, 'LabelVerticalAlignment', 'bottom', ...
          'LabelOrientation', 'horizontal');
end
set(ax, 'XScale', 'log', 'XLim', [20 400], 'XTick', [25 50 100 200 xo], ...
        'XTickLabel', {'25', '50', '100', '200', 'oracle'}, 'YLim', [0.35 1.0]);
xlabel(ax, 'Locally labeled ECGs used for thresholds (k)');
ylabel(ax, 'Macro-F1 on remaining records');
legend(ax, hl, ["External model", "In-population model"], 'Location', 'southeast', 'Box', 'off');
savefig2(f, O, 'Fig7_local_recalibration');
%% Fig. S1 — per-label held-out AUROC for the three objectives (supplement)
P = rd('figS1_perclass_variants.csv');
yl = [floor(100*min(P.auroc))/100 - 0.01, 1];
f = newfig(W2, 2.5);
tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
h = gobjects(1, 3);
for i = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    for v = 1:3
        mu = zeros(1, 6); sd = zeros(1, 6);
        for j = 1:6
            a = P.auroc(P.fold == FOLDS(i) & P.variant == VAR(v) & P.label == LAB(j));
            mu(j) = mean(a); sd(j) = std(a);
        end
        h(v) = errorbar(ax, (1:6) + 0.22*(v - 2), mu, sd, VMK{v}, 'Color', VCOL{v}, ...
                        'MarkerFaceColor', VCOL{v}, 'MarkerSize', 4, 'CapSize', 2, 'LineWidth', 0.8);
    end
    set(ax, 'XTick', 1:6, 'XTickLabel', LAB, 'XLim', [0.5 6.5], 'YLim', yl);
    xtickangle(ax, 45);
    if i == 1, ylabel(ax, 'Test AUROC'); end
    paneltitle(ax, "(" + char('a' + i - 1) + ") Held out: " + FOLDS(i));
end
lg = legend(ax, h, VAR, 'Location', 'southwest', 'Box', 'off');   % last panel: empty lower left
lg.ItemTokenSize = [8 6];
savefig2(f, O, 'FigS1_perlabel_variants');

%% Fig. S2 — worst-group AUROC: source validation vs. held-out cohort (supplement)
G = rd('figS2_worst_group_val_test.csv');
f = newfig(W2, 2.3);
tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    for v = 1:3
        r = G(G.fold == FOLDS(i) & G.variant == VAR(v), :);
        mu = [mean(r.val_worst_group), mean(r.test_worst_group)];
        sd = [std(r.val_worst_group), std(r.test_worst_group)];
        h(v) = errorbar(ax, [1 2] + 0.07*(v - 2), mu, sd, ['-' VMK{v}], 'Color', VCOL{v}, ...
                        'MarkerFaceColor', VCOL{v}, 'MarkerSize', 4, 'CapSize', 2, 'LineWidth', 0.8);
    end
    set(ax, 'XTick', [1 2], 'XTickLabel', {'Source val.', 'Held out'}, 'XLim', [0.6 2.4]);
    if i == 1, ylabel(ax, 'Worst-group AUROC'); end
    paneltitle(ax, "(" + char('a' + i - 1) + ") Held out: " + FOLDS(i));
end
lg = legend(ax, h, VAR, 'Location', 'southwest', 'Box', 'off');
lg.ItemTokenSize = [14 6];
savefig2(f, O, 'FigS2_worst_group_val_test');

fprintf('Figures written to %s\n', O);

%% ---------------------------------------------------------------- helpers
function f = newfig(w, h)
    f = figure('Units', 'inches', 'Position', [1 1 w h], 'Color', 'w');
    % R2025a and later export in the current app theme; force a light figure
    try, f.Theme = 'light'; catch, end
    set(f, 'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 8, ...
           'DefaultTextFontName', 'Arial', 'DefaultTextFontSize', 8, ...
           'DefaultAxesLineWidth', 0.6, 'DefaultAxesTickDir', 'out', ...
           'DefaultAxesTickLength', [0.015 0.015], ...
           'DefaultAxesColor', 'w', 'DefaultAxesXColor', 'k', ...
           'DefaultAxesYColor', 'k', 'DefaultAxesZColor', 'k', ...
           'DefaultTextColor', 'k', 'DefaultAxesGridColor', [0.15 0.15 0.15]);
end

function lightaxes(f)
    % Belt and braces: repaint anything the theme may still have darkened.
    set(findall(f, 'Type', 'axes'), 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    for a = findall(f, 'Type', 'axes')'
        set([a.Title a.XLabel a.YLabel], 'Color', 'k');
    end
    for g = findall(f, 'Type', 'tiledlayout')'   % shared layout labels
        set([g.Title g.XLabel g.YLabel], 'Color', 'k');
    end
    set(findall(f, 'Type', 'colorbar'), 'Color', 'k');
    set(findall(f, 'Type', 'legend'), 'TextColor', 'k', 'Color', 'none');
    set(f, 'Color', 'w');
end

function paneltitle(ax, s)
    title(ax, s, 'FontWeight', 'bold', 'FontSize', 8);
    ax.TitleHorizontalAlignment = 'left';
end

function savefig2(f, O, name)
    lightaxes(f);
    exportgraphics(f, fullfile(O, [name '.pdf']), 'ContentType', 'vector');
    exportgraphics(f, fullfile(O, [name '.png']), 'Resolution', 600);
end

function cm = blues(n)
    t = linspace(0, 1, n)';
    cm = (1 - t) .* [0.97 0.98 1.00] + t .* [0.03 0.19 0.42];
end
