export default function IconButton({
  icon: Icon,
  label,
  onClick,
  disabled = false,
  className = '',
  iconClassName = 'h-5 w-5',
  type = 'button',
  title,
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={title || label}
      className={`inline-flex items-center justify-center ${className}`}
    >
      {Icon ? <Icon className={iconClassName} /> : null}
    </button>
  );
}
