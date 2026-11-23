%% ============================================================
%  TAREFA 3.2 - Sistema CBR Completo
%  Conhecimento e Raciocínio - 2025/2026
%% ============================================================
% PRÉ-REQUISITO: correr primeiro tarefa3_1.m para gerar dataset_tratado.mat

%% --- Carregar dados tratados da tarefa 3.1 ---
load('dataset_tratado.mat'); % carrega: data, X, X_norm, X_min, X_max,
                             %          X_range, targets, nomes, pesos
fprintf('Dataset carregado: %d casos na Case Library.\n', size(X,1));

%% --- Configuração ---
limiar = 0.75; % limiar de similaridade para o Retrieve

%% ============================================================
%  PARTE A - SISTEMA CBR INTERATIVO
%% ============================================================

%% --- TREINAR REDE NEURONAL PARA O REUSE ---
% A rede aprende a relação entre vibration, rotation_speed e voltage
% para propor um valor ajustado de temperature
fprintf('\nA treinar rede neuronal para Reuse...\n');
net_reuse = treinar_rede_reuse(X);
fprintf('Rede treinada.\n');

%% --- LANÇAR SISTEMA CBR INTERATIVO ---
[X, targets] = cbr_sistema(X, X_norm, X_min, X_range, targets, pesos, nomes, limiar, net_reuse);

%% ============================================================
%  PARTE B - TESTAR O SISTEMA CBR
%% ============================================================

%% --- Carregar e tratar dataset de teste ---
fprintf('\n=== A carregar dataset de teste ===\n');
X_test       = carregar_dataset_teste('dataset_TP_test.csv', nomes);
opts_t       = detectImportOptions('dataset_TP_test.csv');
opts_t       = setvartype(opts_t, {'class_cat'}, 'char');
test_tb      = readtable('dataset_TP_test.csv', opts_t);
targets_test = test_tb.class_cat;
fprintf('Dataset de teste carregado: %d casos.\n', size(X_test,1));

%% --- Testar diferentes configurações de pesos ---
fprintf('\n=== TESTE COM DIFERENTES PESOS ===\n');

% Configuração 1: pesos originais (baseados no domínio)
pesos1 = pesos;

% Configuração 2: pesos iguais para todos os atributos
pesos2 = ones(1,14) / 14;

% Configuração 3: foco nos atributos elétricos
pesos3 = [0.05, 0.05, 0.05, 0.20, 0.20, ...
          0.05, 0.05, 0.05, 0.05, 0.05, ...
          0.05, 0.05, 0.05, 0.05];

configs      = {pesos1, pesos2, pesos3};
nomes_config = {'Pesos originais', 'Pesos iguais', 'Foco eletrico'};

fprintf('\n%-20s | %-18s | %-18s\n', 'Configuracao', 'Nao normalizado', 'Normalizado');
fprintf('%s\n', repmat('-', 1, 62));

for c = 1:3
    acc = testar_cbr(X_test, X, X_range, targets, targets_test, configs{c});
    X_test_norm = (X_test - X_min) ./ X_range;
    acc_norm    = testar_cbr(X_test_norm, X_norm, ones(1,14), targets, targets_test, configs{c});
    fprintf('%-20s | %14.2f%%    | %14.2f%%\n', nomes_config{c}, acc, acc_norm);
end

%% ============================================================
%  FUNÇÕES
%% ============================================================

%% --- SISTEMA CBR PRINCIPAL (4 fases) ---
function [X, targets] = cbr_sistema(X, X_norm, X_min, X_range, targets, pesos, nomes, limiar, net)
    fprintf('\n========================================\n');
    fprintf('  SISTEMA CBR - Diagnostico de Motor\n');
    fprintf('========================================\n');

    continuar = true;
    while continuar

        % Pedir os 14 atributos ao utilizador (incluindo temperature)
        novo_caso = pedirNovoCaso(nomes);

        % ---- FASE 1: RETRIEVE ----
        fprintf('\n--- FASE 1: RETRIEVE ---\n');
        [idx_acima, sims] = retrieve(novo_caso, X, X_range, pesos, limiar);
        [~, idx_melhor]   = max(sims);

        if isempty(idx_acima)
            fprintf('Nenhum caso acima do limiar %.2f.\n', limiar);
            fprintf('Caso mais similar: %d | Sim: %.4f | Classe: %s\n', ...
                    idx_melhor, sims(idx_melhor), targets{idx_melhor});
        else
            fprintf('Casos com similaridade > %.2f:\n', limiar);
            for i = 1:length(idx_acima)
                fprintf('  Caso %d | Sim: %.4f | Classe: %s\n', ...
                        idx_acima(i), sims(idx_acima(i)), targets{idx_acima(i)});
            end
        end
        classe_proposta = targets{idx_melhor};
        fprintf('Classe proposta (caso mais similar): %s\n', classe_proposta);

        % ---- FASE 2: REUSE ----
        fprintf('\n--- FASE 2: REUSE ---\n');
        % RN usa vibration(2), rotation_speed(3), voltage(4) para propor temperature ajustada
        entradas_rn   = novo_caso([2, 3, 4])';
        temp_proposta = net(entradas_rn);
        fprintf('Temperature introduzida:      %.4f degC\n', novo_caso(1));
        fprintf('Temperature proposta pela RN: %.4f degC\n', temp_proposta);

        % ---- FASE 3: REVISE ----
        fprintf('\n--- FASE 3: REVISE ---\n');
        fprintf('Aceita a temperature proposta pela rede neuronal? (s/n): ');
        resp_temp = input('', 's');
        if strcmpi(strtrim(resp_temp), 's')
            novo_caso(1) = temp_proposta;
            fprintf('Temperature atualizada para: %.4f degC\n', novo_caso(1));
        else
            fprintf('Temperature mantida: %.4f degC\n', novo_caso(1));
        end
        fprintf('Diagnostico final: %s\n', classe_proposta);

        % ---- FASE 4: RETAIN ----
        fprintf('\n--- FASE 4: RETAIN ---\n');
        fprintf('Adicionar este caso ao dataset? (s/n): ');
        resp_retain = input('', 's');
        if strcmpi(strtrim(resp_retain), 's')
            X       = [X; novo_caso];
            targets = [targets; {classe_proposta}];
            % Recalcular min/max/range e normalizacao com o novo caso
            X_min   = min(X);
            X_range = max(X) - X_min;
            X_range(X_range == 0) = 1;
            X_norm  = (X - X_min) ./ X_range; 
            fprintf('Caso adicionado. Dataset tem agora %d casos.\n', size(X,1));
            % Guardar no ficheiro para persistir entre sessoes
            save('dataset_tratado.mat', 'X', 'X_norm', 'X_min', 'X_range', 'targets');
            fprintf('Dataset guardado em: dataset_tratado.mat\n');
        else
            fprintf('Caso nao adicionado.\n');
        end

        fprintf('\nTestar outro caso? (s/n): ');
        resp = input('', 's');
        continuar = strcmpi(strtrim(resp), 's');
    end
    fprintf('\nSistema CBR terminado.\n');
end

%% --- RETRIEVE ---
function [idx_acima, sims] = retrieve(query, X, X_range, pesos, limiar)
    if nargin < 5, limiar = 0.75; end
    n    = size(X, 1);
    sims = zeros(n, 1);

    for j = 1:n
        sim_local = 1 - abs(query - X(j,:)) ./ X_range;
        sim_local = max(sim_local, 0);
        sims(j)   = sum(pesos .* sim_local) / sum(pesos);
    end

    idx_acima = find(sims >= limiar);
    if ~isempty(idx_acima)
        [~, ord]  = sort(sims(idx_acima), 'descend');
        idx_acima = idx_acima(ord);
    end
end

%% --- TREINAR REDE NEURONAL PARA REUSE ---
function net = treinar_rede_reuse(X)
    % Entradas: vibration(2), rotation_speed(3), voltage(4)
    % Saida:    temperature(1)
    entradas = X(:, [2,3,4])';
    saidas   = X(:, 1)';

    net = feedforwardnet(10);
    net.divideParam.trainRatio = 0.70;
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0.15;
    net.trainParam.showWindow  = false;

    [net, ~] = train(net, entradas, saidas);
end

%% --- TESTAR CBR ---
function acc = testar_cbr(X_test, X_lib, X_range, targets_lib, targets_test, pesos)
    n       = size(X_test, 1);
    acertos = 0;

    for i = 1:n
        sims = zeros(size(X_lib,1), 1);
        for j = 1:size(X_lib,1)
            sim_local = 1 - abs(X_test(i,:) - X_lib(j,:)) ./ X_range;
            sim_local = max(sim_local, 0);
            sims(j)   = sum(pesos .* sim_local) / sum(pesos);
        end

        [~, idx_melhor] = max(sims);
        if strcmpi(strtrim(targets_lib{idx_melhor}), strtrim(targets_test{i}))
            acertos = acertos + 1;
        end
    end

    acc = (acertos / n) * 100;
end

%% --- CARREGAR E CONVERTER DATASET DE TESTE ---
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

%% --- PEDIR NOVO CASO AO UTILIZADOR ---
function novo_caso = pedirNovoCaso(nomes)
    referencias = [
        47.65,  93.13,  68.21;  % temperature      (C)  -- sera ajustada pela RN
         0.28,   4.45,   1.83;  % vibration
      1029.63,1797.31,1465.49;  % rotation_speed   (RPM)
       197.31, 244.58, 226.95;  % voltage          (V)
         7.10,  20.86,  12.37;  % current          (A)
         3.70,   7.66,   5.32;  % pressure         (bar)
        55.85,  93.19,  71.72;  % noise_level      (dB)
         0.59,   0.96,   0.79;  % efficiency       (0-1)
        32.53,  85.12,  56.62;  % load_val
        63.88, 165.23, 112.82;  % torque           (Nm)
         1.00,   3.00,   2.00;  % maintenance_level (1=Low,2=Med,3=High)
         1.00,   3.00,   2.00;  % operating_mode   (1=Idle,2=Normal,3=Overload)
         0.00,   1.00,   1.00;  % cooling_type     (0=Air,1=Oil)
         0.00,   1.00,   1.00;  % sensor_status    (0=OK,1=Warning)
    ];

    fprintf('\nIntroduza os valores do novo caso:\n');
    fprintf('(temperature sera ajustada pela RN na fase de Reuse)\n');
    fprintf('  %-20s | %8s %8s %8s\n', 'Atributo', 'Min', 'Max', 'Media');
    fprintf('  %s\n', repmat('-', 1, 50));

    novo_caso = zeros(1, 14);
    for i = 1:14
        fprintf('  %-20s [min=%.2f, max=%.2f, media=%.2f]: ', ...
                nomes{i}, referencias(i,1), referencias(i,2), referencias(i,3));
        novo_caso(i) = input('');
    end
end