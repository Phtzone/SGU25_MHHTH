
import json

notebook_path = 'seir_dengue_workflow.ipynb'

def modify_notebook_reporting_rate():
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {notebook_path} not found.")
        return

    modified = False
    
    # 1. New code block to redefine functions with Reporting Rate (rho)
    new_code_block = """# --- CẢI TIẾN 1: THÊM REPORTING RATE (RHO) ---
print("UPDATED: Implementing Reporting Rate (rho) support")

# Cần định nghĩa lại build_bounds, default_initial_params và seir_objective

def build_bounds():
    \"\"\"
    Thêm bounds cho rho (tỷ lệ báo cáo), ví dụ từ 5% đến 100%.
    Đã BẬT LẠI Fourier (Seasonality).
    \"\"\"
    fourier_max = config.get('fourier_sum_max', 0.8)
    bounds = [
        tuple(config['beta_intercept_bounds']),  # β0
        tuple(config['beta_feature_bounds']),    # b_rain
        tuple(config['beta_feature_bounds']),    # b_temp
        (0.2, min(0.7, fourier_max - 0.1)),      # α1 (BẬT LẠI: 0.2 -> max)
        (-2*np.pi, 2*np.pi),                     # φ1
        (0.0, min(0.4, fourier_max - 0.2)),      # α2 (BẬT LẠI: 0.0 -> max)
        (-2*np.pi, 2*np.pi),                     # φ2
    ]
    
    # Sigma và Gamma - use fixed if available
    if config.get('fixed_sigma') is not None:
        bounds.append((config['fixed_sigma'], config['fixed_sigma']))
    else:
        bounds.append(tuple(config['sigma_bounds']))
    
    if config.get('fixed_gamma') is not None:
        bounds.append((config['fixed_gamma'], config['fixed_gamma']))
    else:
        bounds.append(tuple(config['gamma_bounds']))
        
    # --- THAM SỐ MỚI ---
    bounds.append((0.05, 1.0))  # Rho (Reporting Rate): 5% -> 100%
    
    return bounds


def default_initial_params():
    \"\"\"Thêm giá trị khởi tạo cho rho = 0.2 (20%)\"\"\"
    beta0 = np.clip(np.log(0.6), config['beta_intercept_bounds'][0], config['beta_intercept_bounds'][1])
    b_rain = 0.1
    b_temp = 0.1
    alpha1 = 0.5
    phi1 = -2 * np.pi * 8 / 12
    alpha2 = 0.2
    phi2 = 0.0
    
    # Fixed parameters nếu có
    if config.get('fixed_sigma') is not None:
        sigma = config['fixed_sigma']
    else:
        sigma = np.mean(config['sigma_bounds'])
    
    if config.get('fixed_gamma') is not None:
        gamma = config['fixed_gamma']
    else:
        gamma = np.mean(config['gamma_bounds'])
    
    # Tạo mảng params cơ bản
    params = [beta0, b_rain, b_temp, alpha1, phi1, alpha2, phi2, sigma, gamma]
    
    # Thêm Rho init
    params.append(0.2) 
    
    return np.array(params)


def seir_objective(params, initial_state, populations, rain_data, temp_data, t_months, observed_target):
    \"\"\"
    Cập nhật hàm mục tiêu để nhân với Rho (Reporting Rate).
    Mô hình dự báo số ca nhiễm THỰC TẾ, sau đó nhân với Rho để ra số ca BÁO CÁO.
    Loss tính trên số ca báo cáo này.
    \"\"\"
    try:
        # Tách rho ra khỏi params (phần tử cuối cùng)
        rho = params[-1]
        seir_params = params[:-1] # 9 tham số đầu là cho SEIR (beta, alpha, sigma, gamma...)

        # Tighter constraint: α1 + α2 < fourier_sum_max (default 0.8)
        fourier_max = config.get('fourier_sum_max', 0.8)
        if seir_params[3] + seir_params[5] >= fourier_max:
            return 1e10
        
        # Chạy mô hình SEIR (trả về TỔNG số ca nhiễm thực tế trong cộng đồng)
        sim_incidence_true, _ = simulate_seir(seir_params, initial_state.copy(), populations, rain_data, temp_data, t_months)
        
        # Ca báo cáo = Ca thực tế * Rho
        # Đây là bước quan trọng: Scale down số ca thực tế để so sánh với dữ liệu giám sát
        sim_reported = sim_incidence_true * rho 
        
        # Áp dụng transform (per_capita, zscore...) lên sim_reported
        sim_target = transform_cases_to_target(sim_reported, populations)

        if config['loss_on_log_scale']:
            sim_vals = np.log(sim_target + config['loss_epsilon'])
            obs_vals = np.log(observed_target + config['loss_epsilon'])
        else:
            sim_vals = sim_target
            obs_vals = observed_target

        residuals = sim_vals - obs_vals
        
        # Standard MSE (hoặc Weighted MSE)
        loss = weighted_mse(residuals, observed_target)
        
        if not np.isfinite(loss):
            return 1e10
        return loss
    except Exception:
        return 1e10

def sample_random_params(bounds=None):
    \"\"\"Sample random parameters updated for rho\"\"\"
    if bounds is None:
        bounds = build_bounds()
    
    fourier_max = config.get('fourier_sum_max', 0.8)
    
    params = []
    for i, bound in enumerate(bounds):
        if bound[0] == bound[1]:
            params.append(bound[0])
        else:
            params.append(np.random.uniform(bound[0], bound[1]))
    
    # Ensure Fourier constraint (chỉ check nếu có đủ tham số)
    if len(params) >= 6:
        alpha1 = params[3]
        alpha2 = params[5]
        if alpha1 + alpha2 >= fourier_max:
            params[5] = max(bounds[5][0], min(bounds[5][1], fourier_max - alpha1 - 0.01))
    
    return np.array(params)

print("✅ Functions updated: Reporting Rate (rho) integrated.")
"""

    # Find the cell to replace (Cell 7 based on previous interactions)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Look for the signature of the functions we want to replace
            if "def build_bounds():" in source and "def default_initial_params():" in source and "def estimate_initial_state" in source:
                print("Found target cell (Cell 7). Replacing content...")
                # Replace content
                cell['source'] = [line + '\n' for line in new_code_block.split('\n')]
                if cell['source'][-1] == '\n':
                    cell['source'].pop()
                modified = True
                break
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Successfully modified notebook to include Reporting Rate.")
    else:
        print("Could not find the target cell. Please check notebook structure.")

if __name__ == "__main__":
    modify_notebook_reporting_rate()

