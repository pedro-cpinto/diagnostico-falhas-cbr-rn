%% ============================================================
%  TAREFA 3.3 - Redes Neuronais Feedforward
%  Conhecimento e Raciocínio - 2025/2026
%% ============================================================
% PRÉ-REQUISITO: correr primeiro tarefa3_1.m para gerar dataset_tratado.mat

load('dataset_tratado.mat'); % carrega: X, X_norm, targets, nomes, X_min, X_range

%% --- Parte i) Construir matriz de entrada ---
% X já está construída em tarefa3_1.m (5000 x 14)
% Transpor para formato MATLAB: (14 atributos) x (N exemplos)
Xtr   = X';       % não normalizado  → (14 x 5000)
Xtr_n = X_norm';  % normalizado      → (14 x 5000)

%% --- Parte ii) Construir target binário ---
% MATLAB feedforwardnet espera: cada COLUNA = um exemplo, cada LINHA = uma classe
% 3 classes → matriz T de dimensão (3 x N)
% Linha 1: Normal | Linha 2: ElectricalFailure | Linha 3: MechanicalFailure
classes = {'Normal', 'ElectricalFailure', 'MechanicalFailure'};

T = zeros(3, length(targets));
for i = 1:length(targets)
    t = strtrim(targets{i});
    if     strcmpi(t, 'Normal'),            T(1,i) = 1;
    elseif strcmpi(t, 'ElectricalFailure'), T(2,i) = 1;
    else,                                   T(3,i) = 1;  % MechanicalFailure
    end
end
fprintf('Target binário construído: %d exemplos, %d classes\n', size(T,2), size(T,1));
fprintf('  Normal: %d | ElectricalFailure: %d | MechanicalFailure: %d\n', ...
        sum(T(1,:)), sum(T(2,:)), sum(T(3,:)));

% %% --- Parte iii) Configurações a testar ---
% % Topologias: [neurónios por camada]
% %   [10]      = 1 camada oculta, 10 neurónios (default-like)
% %   [20]      = 1 camada oculta, 20 neurónios
% %   [10 10]   = 2 camadas ocultas, 10+10
% %   [20 10]   = 2 camadas ocultas, 20+10
% topologias = {[10], [20], [10 10], [20 10]};
% 
% % Funções de treino (pedidas no enunciado)
% %   trainlm  = Levenberg-Marquardt (rápido, bom para datasets pequenos/médios)
% %   trainbr  = Bayesian Regularization (evita overfitting, mais lento)
% %   traingd  = Gradient Descent simples (mais lento, referência)
% %   trainscg = Scaled Conjugate Gradient (bom equilíbrio velocidade/precisão)
% f_treinos = {'trainlm', 'trainbr', 'traingd', 'trainscg'};
% 
% % Funções de ativação na camada de SAÍDA (pedidas no enunciado)
% %   logsig  = sigmoid logística, saída [0,1], boa para classificação binária
% %   softmax = normaliza saídas para somarem 1 (probabilidades), ideal para multiclasse
% %   tansig  = tangente hiperbólica, saída [-1,1]
% %   purelin = linear, raramente ideal para classificação
% f_ativacoes = {'logsig', 'softmax', 'tansig', 'purelin'};
% 
% % Divisões treino/validação/teste
% divisoes = {[0.70 0.15 0.15], [0.80 0.10 0.10]};

topologias  = {[10], 20, [10 10], [20 20]};
f_treinos   = {'trainlm', 'trainbr', 'traingd', 'trainscg'};
f_ativacoes = {'logsig', 'softmax', 'tansig', 'purelin'};
divisoes    = {[0.70 0.15 0.15], [0.80 0.10 0.10]};
% Taxa de aprendizagem — só usada com traingd (os outros algoritmos
% adaptam a lr automaticamente e ignoram este parâmetro)
lr_values   = [0.01, 0.05, 0.1];

n_reps = 10; % repetições por configuração (para obter média estável)

%% --- Loop principal: treinar e avaliar todas as configurações ---
% lr_values só se aplica ao traingd; os restantes algoritmos têm 1 valor de lr
n_lr_traingd = length(lr_values);  % 3 valores de lr para traingd
n_lr_outros  = 1;                  % 1 valor (irrelevante) para os restantes
n_treinos_sem_lr = (length(f_treinos) - 1) * n_lr_outros;  % trainlm, trainbr, trainscg
n_treinos_traingd = 1 * n_lr_traingd;                       % traingd × lr
n_configs = length(topologias) * (n_treinos_sem_lr + n_treinos_traingd) * length(f_ativacoes) * length(divisoes);
fprintf('\nTotal de configurações: %d (x%d reps = %d treinos)\n', ...
        n_configs, n_reps, n_configs * n_reps);

% Pré-alocar matrizes de resultados
resultados_global     = zeros(n_configs, 1);
resultados_teste      = zeros(n_configs, 1);
resultados_global_n   = zeros(n_configs, 1); % normalizado
resultados_teste_n    = zeros(n_configs, 1); % normalizado
config_labels         = cell(n_configs, 1);
todas_configs         = cell(n_configs, 1);

idx_config = 0;

for t = 1:length(topologias)
    for f = 1:length(f_treinos)

        % Definir lista de lr a testar:
        % traingd usa lr fixa → testar vários valores
        % os restantes adaptam lr automaticamente → usar só 0 (ignorado)
        if strcmp(f_treinos{f}, 'traingd')
            lrs = lr_values;
        else
            lrs = 0; % valor ignorado por trainlm, trainbr, trainscg
        end

        for lr_idx = 1:length(lrs)
            lr = lrs(lr_idx);

            for e = 1:length(f_ativacoes)
                for d = 1:length(divisoes)

                    topologia  = topologias{t};
                    f_treino   = f_treinos{f};
                    f_ativacao = f_ativacoes{e};
                    divisao    = divisoes{d};

                    acc_global   = zeros(1, n_reps);
                    acc_teste    = zeros(1, n_reps);
                    acc_global_n = zeros(1, n_reps);
                    acc_teste_n  = zeros(1, n_reps);

                    for r = 1:n_reps

                        % --- NÃO NORMALIZADO ---
                        net = construir_rede(topologia, f_treino, f_ativacao, divisao, lr);
                        [net, tr] = train(net, Xtr, T);
                        acc_global(r) = calcular_acerto(net(Xtr), T);
                        acc_teste(r)  = calcular_acerto(net(Xtr(:, tr.testInd)), T(:, tr.testInd));

                        % --- NORMALIZADO ---
                        net_n = construir_rede(topologia, f_treino, f_ativacao, divisao, lr);
                        [net_n, tr_n] = train(net_n, Xtr_n, T);
                        acc_global_n(r) = calcular_acerto(net_n(Xtr_n), T);
                        acc_teste_n(r)  = calcular_acerto(net_n(Xtr_n(:, tr_n.testInd)), T(:, tr_n.testInd));

                    end % r

                    idx_config = idx_config + 1;
                    resultados_global(idx_config)   = mean(acc_global);
                    resultados_teste(idx_config)    = mean(acc_teste);
                    resultados_global_n(idx_config) = mean(acc_global_n);
                    resultados_teste_n(idx_config)  = mean(acc_teste_n);

                    % Label: incluir lr só quando relevante (traingd)
                    if strcmp(f_treino, 'traingd')
                        label = sprintf('Top:%s|%s|lr=%.2f|%s|%.0f/%.0f/%.0f', ...
                            mat2str(topologia), f_treino, lr, f_ativacao, ...
                            divisao(1)*100, divisao(2)*100, divisao(3)*100);
                    else
                        label = sprintf('Top:%s|%s|%s|%.0f/%.0f/%.0f', ...
                            mat2str(topologia), f_treino, f_ativacao, ...
                            divisao(1)*100, divisao(2)*100, divisao(3)*100);
                    end
                    config_labels{idx_config} = label;
                    todas_configs{idx_config} = {topologia, f_treino, f_ativacao, divisao, lr};

                    fprintf('%-65s | NNorm: Global=%.1f%% Teste=%.1f%% | Norm: Global=%.1f%% Teste=%.1f%%\n', ...
                        label, ...
                        mean(acc_global), mean(acc_teste), ...
                        mean(acc_global_n), mean(acc_teste_n));

                end % d
            end % e
        end % lr_idx
    end % f
end % t

fprintf('\nTodos os testes concluídos.\n');

%% --- Identificar 3 melhores e 3 piores (por acc de teste, não normalizado) ---
[~, ord] = sort(resultados_teste, 'descend');
idx_melhores = ord(1:3);
idx_piores   = ord(end-2:end);

fprintf('\n=== 3 MELHORES CONFIGURAÇÕES (por acc teste, não normalizado) ===\n');
for i = 1:3
    fprintf('%d. %s\n   Não-norm: Global=%.2f%% | Teste=%.2f%%\n   Norm:     Global=%.2f%% | Teste=%.2f%%\n', ...
        i, config_labels{idx_melhores(i)}, ...
        resultados_global(idx_melhores(i)), resultados_teste(idx_melhores(i)), ...
        resultados_global_n(idx_melhores(i)), resultados_teste_n(idx_melhores(i)));
end

fprintf('\n=== 3 PIORES CONFIGURAÇÕES ===\n');
for i = 1:3
    fprintf('%d. %s\n   Não-norm: Global=%.2f%% | Teste=%.2f%%\n   Norm:     Global=%.2f%% | Teste=%.2f%%\n', ...
        i, config_labels{idx_piores(i)}, ...
        resultados_global(idx_piores(i)), resultados_teste(idx_piores(i)), ...
        resultados_global_n(idx_piores(i)), resultados_teste_n(idx_piores(i)));
end

%% --- Parte b) plotconfusion: 3 melhores e 3 piores, normalizado vs não normalizado ---
idx_analise = unique([idx_melhores(:); idx_piores(:)], 'stable');

for i = 1:length(idx_analise)
    cfg        = todas_configs{idx_analise(i)};
    topologia  = cfg{1};
    f_treino   = cfg{2};
    f_ativacao = cfg{3};
    divisao    = cfg{4};
    lr         = cfg{5};

    % Determinar rótulo (melhor ou pior)
    if any(idx_melhores == idx_analise(i))
        tipo = sprintf('Melhor%d', find(idx_melhores == idx_analise(i)));
    else
        tipo = sprintf('Pior%d', find(idx_piores == idx_analise(i)));
    end

    % Treinar com NÃO NORMALIZADO
    net_nao = construir_rede(topologia, f_treino, f_ativacao, divisao, lr);
    [net_nao, ~] = train(net_nao, Xtr, T);
    fig = figure('Visible','off');
    plotconfusion(T, net_nao(Xtr));
    title(sprintf('%s - Não Normalizado | %s', tipo, config_labels{idx_analise(i)}), ...
          'Interpreter', 'none');
    saveas(fig, sprintf('confusion_%s_%s_%s_%s_NaoNorm.png',tipo, mat2str(topologia), f_treino, f_ativacao));
    close(fig);

    % Treinar com NORMALIZADO
    net_norm = construir_rede(topologia, f_treino, f_ativacao, divisao, lr);
    [net_norm, ~] = train(net_norm, Xtr_n, T);
    fig = figure('Visible','off');
    plotconfusion(T, net_norm(Xtr_n));
    title(sprintf('%s - Normalizado | %s', tipo, config_labels{idx_analise(i)}), ...
          'Interpreter', 'none');
    saveas(fig, sprintf('confusion_%s_%s_%s_%s_Norm.png', tipo, mat2str(topologia), f_treino, f_ativacao));
    close(fig);

    fprintf('plotconfusion guardado para: %s\n', tipo);

    % Gravar as 3 melhores redes (não normalizadas)
    if any(idx_melhores == idx_analise(i))
        num_melhor = find(idx_melhores == idx_analise(i));
        net = net_nao;
        save(sprintf('melhor_rede_%d.mat', num_melhor), 'net');
        fprintf('Rede melhor_%d gravada.\n', num_melhor);
    end
end

%% --- Parte d) Testar as 3 melhores no dataset de teste externo ---
fprintf('\n=== PARTE d) TESTE NO DATASET EXTERNO ===\n');

% Carregar dataset de teste
X_test = carregar_dataset_teste('dataset_TP_test.csv', nomes);
X_test_n = (X_test - X_min) ./ X_range; % normalizar com os parâmetros do treino

opts_t       = detectImportOptions('dataset_TP_test.csv');
opts_t       = setvartype(opts_t, {'class_cat'}, 'char');
test_tb      = readtable('dataset_TP_test.csv', opts_t);
targets_test = test_tb.class_cat;

% Construir target de teste binário
T_test = zeros(3, length(targets_test));
for i = 1:length(targets_test)
    t = strtrim(targets_test{i});
    if     strcmpi(t, 'Normal'),            T_test(1,i) = 1;
    elseif strcmpi(t, 'ElectricalFailure'), T_test(2,i) = 1;
    else,                                   T_test(3,i) = 1;  % MechanicalFailure
    end
end

fprintf('Dataset de teste: %d exemplos\n', size(X_test,1));
fprintf('  Normal: %d | Electrical: %d | Mechanical: %d\n', ...
        sum(T_test(1,:)), sum(T_test(2,:)), sum(T_test(3,:)));

% Testar cada uma das 3 melhores redes
for i = 1:3
    load(sprintf('melhor_rede_%d.mat', i), 'net');

    Y_test = net(X_test');  % X_test é (30x14), precisa de (14x30)
    acc    = calcular_acerto(Y_test, T_test);

    % Acerto por classe
    [~, pred] = max(Y_test);
    [~, real] = max(T_test);
    for c = 1:3
        idx_c     = (real == c);
        acc_c     = mean(pred(idx_c) == c) * 100;
        fprintf('  Melhor rede %d | Classe %s: %.1f%%\n', i, classes{c}, acc_c);
    end

    fig = figure('Visible','off');
    plotconfusion(T_test, Y_test);
    title(sprintf('Melhor Rede %d - Dataset Teste (acc=%.1f%%)', i, acc), ...
          'Interpreter', 'none');
    saveas(fig, sprintf('confusion_melhorRede%d_teste.png', i));
    close(fig);

    fprintf('Melhor rede %d no dataset de teste: %.2f%%\n', i, acc);
end

fprintf('\nTarefa 3.3 concluída.\n');

%% ============================================================
%  FUNÇÕES AUXILIARES
%% ============================================================

%% --- Construir rede ---
function net = construir_rede(topologia, f_treino, f_ativacao, divisao, lr)
    net = feedforwardnet(topologia, f_treino);
    net.layers{end}.transferFcn    = f_ativacao;
    net.divideParam.trainRatio     = divisao(1);
    net.divideParam.valRatio       = divisao(2);
    net.divideParam.testRatio      = divisao(3);
    net.trainParam.showWindow      = false;
    net.trainParam.showCommandLine = false;
    % Épocas: trainbr é mais lento, limitar para não demorar demasiado
    if strcmp(f_treino, 'trainbr')
        net.trainParam.epochs = 100;
    else
        net.trainParam.epochs = 1000;
    end
    % Taxa de aprendizagem: só relevante para traingd
    % trainlm, trainbr e trainscg adaptam lr automaticamente
    if strcmp(f_treino, 'traingd') && lr > 0
        net.trainParam.lr = lr;
    end
end

%% --- Calcular acerto (%) ---
function acc = calcular_acerto(Y, T)
    % Y e T: (3 x N) — cada coluna é um exemplo
    [~, pred] = max(Y);
    [~, real] = max(T);
    acc = mean(pred == real) * 100;
end

%% --- Carregar e converter dataset de teste ---
function X_test = carregar_dataset_teste(ficheiro, nomes)
    opts = detectImportOptions(ficheiro);
    opts = setvartype(opts, {'maintenance_level','operating_mode', ...
                             'cooling_type','sensor_status','class_cat'}, 'char');
    dt = readtable(ficheiro, opts);

    for i = 1:height(dt)
        switch lower(strtrim(dt.maintenance_level{i}))
            case 'low',    dt.maintenance_level{i} = '1';
            case 'medium', dt.maintenance_level{i} = '2';
            case 'high',   dt.maintenance_level{i} = '3';
        end
        switch lower(strtrim(dt.operating_mode{i}))
            case 'idle',     dt.operating_mode{i} = '1';
            case 'normal',   dt.operating_mode{i} = '2';
            case 'overload', dt.operating_mode{i} = '3';
        end
        switch lower(strtrim(dt.cooling_type{i}))
            case 'air', dt.cooling_type{i} = '0';
            case 'oil', dt.cooling_type{i} = '1';
        end
        switch lower(strtrim(dt.sensor_status{i}))
            case 'ok',      dt.sensor_status{i} = '0';
            case 'warning', dt.sensor_status{i} = '1';
        end
    end
    dt.maintenance_level = str2double(dt.maintenance_level);
    dt.operating_mode    = str2double(dt.operating_mode);
    dt.cooling_type      = str2double(dt.cooling_type);
    dt.sensor_status     = str2double(dt.sensor_status);

    X_test = zeros(height(dt), 14);
    for i = 1:14
        X_test(:,i) = dt.(nomes{i});
    end
end